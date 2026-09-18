"""Phase analysis of corrected STM topography FFT (r3 Bragg peak masks).

The complex FFT2 comes from ``stm_data_processing.utils.bragg_peak.compute_fft2``
and the six r3 peak positions plus their sub-pixel q from the package detection
(``detect_bragg_peaks`` with ``patch_half=8``, i.e. the 17x17 Gaussian localizer);
masks, iFFTs and the Kekule phase construction are unchanged.

Analysis steps:
  1. nomask  - the complex FFT2 itself, plotted in q space
  2. r3all   - mask on all six r3 (inner-ring) peaks, mask and complement
               iFFT -> amplitude (log) + phase (seismic) plots and histograms
  3. Per-peak Kekule phase analysis for ALL six r3 peaks:
       - subpixel peak position via 2D Gaussian fit on the FFT magnitude
         (exact q; NO global plane correction)
       - single-peak circular mask (radius = pct% of image size)
       - complex iFFT -> psi(r) = Delta exp(i(q.r + phi))
       - phi(r) = angle(psi) - q.r  (mod 2pi)
       - amplitude-weighted phi histogram (360 bins) with Z3 markers
       - phi(r) phase map (hsv)
     Outputs: per-peak amplitude/phase plots + npy, per-peak phi map and
     histogram, plus 2x3 summary figures of all six phi histograms and maps.

Usage:
    cd /path/to/STM_DataProcessing
    uv run python skills/stm-topo-phase-analysis/scripts/stm_phase_analysis.py \
        CORRECTED.csv -o OUT_DIR [--fft2 FFT2.npy] [--pct 5] [--bins 1024]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

matplotlib.rcParams.update({
    "text.usetex": False,  # usetex broken with mpl 3.10 + TeX Live 2026
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

BAD_COLOR = "#b0b0b0"
# Sub-pixel Gaussian patch half width of the package localizer: 8 -> 17x17 pixels.
PATCH_HALF = 8
Z3_ANGLES = [0.0, 2 * np.pi / 3, 4 * np.pi / 3]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("input", help="corrected topography CSV (square, may contain NaN)")
    p.add_argument("-o", "--outdir", default=None,
                   help="output directory (default: next to the input CSV)")
    p.add_argument("--fft2", default=None,
                   help="complex FFT2 npy; default: recompute from the CSV with the "
                        "package compute_fft2 (NaN filled with the plane, Hanning "
                        "window, fftshift, no plane subtraction)")
    p.add_argument("-L", "--size-nm", type=float, default=100.0,
                   help="field of view of the corrected image in nm (default 100); used "
                        "by the package detection and labelling (the FFT itself does not "
                        "depend on it)")
    p.add_argument("--pct", type=float, default=5,
                   help="mask radius as percentage of image size (default 5)")
    p.add_argument("--bins", type=int, default=1024,
                   help="number of bins for the 0/pi histograms (default 1024)")
    p.add_argument("--stm-lib", default="/Users/hunfen/Documents/GitHub/STM_DataProcessing/src",
                   help="STM_DataProcessing src directory (for the gwyddion colormap)")
    return p.parse_args()


def r3_peaks(detection):
    """Six r3 (first-ring) peaks of a package detection, 12 o'clock first.

    Returns ``[(integer x, integer y, qx_px, qy_px), ...]``: the integer positions
    centre the analysis masks, the sub-pixel q (17x17 Gaussian localizer inside
    ``detect_bragg_peaks``, ``patch_half=8``) drives the Kekule phase and is used
    as-is - no plane correction is applied to q.  The r3 family is the group of
    peaks labelled ``h^2 + k^2 + hk == 1``; when the lattice is not labelled the six
    innermost detected peaks are used instead (the same set, mirrors included).
    """
    n = detection.n_px
    centre = n // 2
    ring = [
        peak
        for peak in detection.peaks
        if peak.index_hk is not None
        and peak.index_hk[0] ** 2 + peak.index_hk[1] ** 2
        + peak.index_hk[0] * peak.index_hk[1] == 1
    ]
    if len(ring) < 6:
        ring = sorted(detection.peaks, key=lambda peak: float(np.hypot(*peak.q_px)))[:6]
    ring = sorted(ring, key=lambda peak: peak.q_px[1])  # qy ascending -> 12 o'clock
    return [
        (round(float(peak.q_px[0])) + centre, round(float(peak.q_px[1])) + centre,
         float(peak.q_px[0]), float(peak.q_px[1]))
        for peak in ring
    ]


def circle_mask(shape, peaks, radius):
    m = np.zeros(shape, dtype=bool)
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    for px, py in peaks:
        m |= (xx - px) ** 2 + (yy - py) ** 2 <= radius ** 2
    return m


def complex_ifft(freq):
    return np.fft.ifft2(np.fft.ifftshift(freq))


def plot_fft_amp_phase(c, title, outpath):
    """Q-space plot of the FFT2 itself: amplitude (log, inferno) + phase."""
    amp = np.abs(c)
    ph = np.angle(c)
    p_lo, p_hi = np.percentile(amp, [5, 99.5])
    log_amp = np.log(1 + amp)
    norm = np.clip((log_amp - np.log(1 + p_lo)) / (np.log(1 + p_hi) - np.log(1 + p_lo)), 0, 1)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    im0 = axs[0].imshow(norm, cmap="inferno", origin="lower")
    axs[0].set_title(f"{title} - Amplitude (log)", fontsize=13)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].axis("off")
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    im1 = axs[1].imshow(ph, cmap="seismic", origin="lower",
                        vmin=-np.pi, vmax=np.pi)
    axs[1].set_title(f"{title} - Phase", fontsize=13)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].axis("off")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pair(c, valid, title, outpath, gwyddion):
    """Real-space plot: amplitude (log, gwyddion) + phase (seismic)."""
    amp = np.where(valid, np.abs(c), np.nan)
    ph = np.where(valid, np.angle(c), np.nan)
    v = np.abs(c)[valid]
    cmap_amp = gwyddion.copy()
    cmap_amp.set_bad(color=BAD_COLOR)
    cmap_ph = plt.get_cmap("seismic").copy()
    cmap_ph.set_bad(color=BAD_COLOR)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    if v.max() <= 0:
        im0 = axs[0].imshow(amp, cmap=cmap_amp, origin="lower", vmin=0, vmax=1)
    else:
        lin = v[v > 0].min()
        norm = SymLogNorm(linthresh=lin, linscale=0.5, vmin=0.0, vmax=v.max())
        im0 = axs[0].imshow(amp, cmap=cmap_amp, origin="lower", norm=norm)
    axs[0].set_title(f"{title} - Amplitude (log)", fontsize=13)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].axis("off")
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    im1 = axs[1].imshow(ph, cmap=cmap_ph, origin="lower",
                        vmin=-np.pi, vmax=np.pi)
    axs[1].set_title(f"{title} - Phase", fontsize=13)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].axis("off")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def kekule_phi(psi, q_px, topo):
    """phi(r) = angle(psi) - q.r (mod 2pi) with the exact subpixel q."""
    n = topo.shape[0]
    yy, xx = np.mgrid[:n, :n]
    xc = yc = n // 2
    q_dot_r = (2 * np.pi / n) * (q_px[0] * (xx - xc) + q_px[1] * (yy - yc))
    return np.mod(np.angle(psi) - q_dot_r, 2 * np.pi)


def plot_kekule_phi(phi_field, good, amp, title, outpath):
    """Per-peak figure: phi(r) map (hsv) + amplitude-weighted histogram."""
    bins = np.linspace(0, 2 * np.pi, 361)
    h, edges = np.histogram(phi_field[good], bins=bins,
                            weights=amp[good], density=True)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    im = axs[0].imshow(np.where(good, phi_field, np.nan), cmap="hsv",
                       origin="lower", vmin=0, vmax=2 * np.pi)
    axs[0].set_title(f"{title} - phi(r) map", fontsize=13)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)
    axs[1].plot((edges[:-1] + edges[1:]) / 2, h, color="steelblue", lw=1.5)
    for z in Z3_ANGLES:
        axs[1].axvline(z, color="red", ls="--", lw=1.2)
    axs[1].set_xlabel("phi (rad)", fontsize=13)
    axs[1].set_ylabel("amp-weighted density", fontsize=13)
    axs[1].set_title(f"{title} - phi distribution", fontsize=13)
    axs[1].set_xticks([0, np.pi / 3, 2 * np.pi / 3, np.pi,
                       4 * np.pi / 3, 5 * np.pi / 3, 2 * np.pi])
    axs[1].set_xticklabels(["0", "pi/3", "2pi/3", "pi", "4pi/3", "5pi/3", "2pi"])
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return h, edges


def hist_phase(c, valid, use_valid):
    amp = np.abs(c)
    ph = np.angle(c)
    if use_valid:
        v = amp[valid]
        thresh = np.percentile(v, 10) if v.max() > 0 else 0.0
        good = valid & (amp > thresh)
    else:
        good = np.ones(amp.shape, dtype=bool)
    return ph[good]


def triple_product_combo(q_by_name):
    """Pick the three inequivalent peaks whose q vectors sum closest to zero."""
    import itertools

    best, best_sum = None, float("inf")
    for combo in itertools.combinations(q_by_name, 3):
        s = np.sum([q_by_name[c] for c in combo], axis=0)
        if np.linalg.norm(s) < best_sum:
            best_sum = np.linalg.norm(s)
            best = combo
    return best


def combined_theta(P, q_sum, topo):
    """theta(r) = angle(P) - q_sum.r (mod 2pi); the slow Z3 order-parameter
    phase built from the triple product of three inequivalent components."""
    n = topo.shape[0]
    xc = yc = n // 2
    yy, xx = np.mgrid[:n, :n]
    q_sum_dot_r = (2 * np.pi / n) * (q_sum[0] * (xx - xc) + q_sum[1] * (yy - yc))
    return np.mod(np.angle(P) - q_sum_dot_r, 2 * np.pi)


def plot_combined(theta, good, amp, combo, outpath):
    """Combined Kekule phase figure: theta(r) map + Z3-marked histogram."""
    bins = np.linspace(0, 2 * np.pi, 361)
    h, edges = np.histogram(theta[good], bins=bins,
                            weights=amp[good], density=True)
    combo_str = "+".join(c[-2:] for c in combo)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    im = axs[0].imshow(np.where(good, theta, np.nan), cmap="hsv",
                       origin="lower", vmin=0, vmax=2 * np.pi)
    axs[0].set_title(f"combined theta(r) = phi({combo_str})", fontsize=13)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)
    axs[1].plot((edges[:-1] + edges[1:]) / 2, h, color="steelblue", lw=1.5)
    for z in Z3_ANGLES:
        axs[1].axvline(z, color="red", ls="--", lw=1.2)
    axs[1].set_xlabel("theta (rad)", fontsize=13)
    axs[1].set_ylabel("amp-weighted density", fontsize=13)
    axs[1].set_title("combined Kekule phase distribution", fontsize=13)
    axs[1].set_xticks([0, np.pi / 3, 2 * np.pi / 3, np.pi,
                       4 * np.pi / 3, 5 * np.pi / 3, 2 * np.pi])
    axs[1].set_xticklabels(["0", "pi/3", "2pi/3", "pi", "4pi/3", "5pi/3", "2pi"])
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    for z in Z3_ANGLES:
        idx = np.argmin(np.abs((edges[:-1] + edges[1:]) / 2 - z))
        print(f"    combined Z3 {np.degrees(z):.0f}deg: {h[idx]:.4f}")
    print(f"    combined bg median: {np.median(h):.4f}")
    return h, edges


def main():
    args = parse_args()
    sys.path.insert(0, args.stm_lib)
    from stm_data_processing.stm.preview_plot import gwyddion
    from stm_data_processing.utils.bragg_peak import compute_fft2, detect_bragg_peaks

    csv_path = Path(args.input)
    outdir = Path(args.outdir) if args.outdir else csv_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    topo = np.loadtxt(csv_path, delimiter=",")
    valid = ~np.isnan(topo)
    n = topo.shape[0]
    radius = int(args.pct / 100 * n)

    fft2 = (np.load(args.fft2) if args.fft2
            else compute_fft2(topo, args.size_nm, subtract_plane=False))
    print(f"FFT2: {fft2.shape}, {fft2.dtype}")

    detection = detect_bragg_peaks(topo, args.size_nm, patch_half=PATCH_HALF,
                                   subtract_plane=False, return_fft2=False)
    inner = r3_peaks(detection)
    print(f"lattice: {detection.meta.get('basis_source')}, "
          f"{len(detection.peaks)} peaks reported, "
          f"{len(inner)} r3 peaks at 12 o'clock first: "
          f"{[(px, py) for px, py, _, _ in inner]}")

    # --- 1. nomask: the FFT2 itself in q space ---
    plot_fft_amp_phase(fft2, "nomask (FFT2)", outdir / "nomask.png")
    print(f"nomask: FFT2 amplitude max={np.abs(fft2).max():.4e}")

    # --- 2. r3all: mask on all six r3 peaks, mask + complement ---
    m = circle_mask(fft2.shape, [(px, py) for px, py, _, _ in inner], radius)
    c_in = complex_ifft(np.where(m, fft2, 0.0))
    c_out = complex_ifft(np.where(~m, fft2, 0.0))
    np.save(outdir / "r3all_mask_in_ifft.npy", c_in)
    np.save(outdir / "r3all_mask_out_ifft.npy", c_out)
    plot_pair(c_in, valid, "r3all mask IN", outdir / "r3all_in.png", gwyddion)
    plot_pair(c_out, valid, "r3all mask OUT", outdir / "r3all_out.png", gwyddion)
    print(f"r3all: in amp max={np.abs(c_in).max():.4e}, "
          f"out amp max={np.abs(c_out).max():.4e}")

    # --- 3. per-peak Kekule phase analysis for all six r3 peaks ---
    phi_fields = {}
    histograms = {}
    q_list = []
    q_by_name = {}
    psi_by_name = {}
    for i, (px0, py0, qx, qy) in enumerate(inner):
        name = f"r3p{i}"
        q_list.append((qx, qy))
        q_by_name[name] = (qx, qy)
        print(f"{name}: integer=({px0},{py0}) q=({qx:.3f},{qy:.3f}) px")

        # single-peak mask, complex iFFT
        m = circle_mask(fft2.shape, [(px0, py0)], radius)
        c_in = complex_ifft(np.where(m, fft2, 0.0))
        c_out = complex_ifft(np.where(~m, fft2, 0.0))
        psi_by_name[name] = c_in
        np.save(outdir / f"{name}_mask_in_ifft.npy", c_in)
        np.save(outdir / f"{name}_mask_out_ifft.npy", c_out)
        plot_pair(c_in, valid, f"{name} mask IN",
                  outdir / f"{name}_in.png", gwyddion)
        plot_pair(c_out, valid, f"{name} mask OUT",
                  outdir / f"{name}_out.png", gwyddion)

        # Kekule phase: exact q, no plane correction
        phi_field = kekule_phi(c_in, (qx, qy), topo)
        amp = np.abs(c_in)
        thresh = np.percentile(amp[valid], 50)
        good = valid & (amp > thresh)
        phi_fields[name] = phi_field
        h, edges = plot_kekule_phi(phi_field, good, amp, f"{name} Kekule",
                                   outdir / f"{name}_kekule_phi.png")
        histograms[name] = (h, edges)
        z_dens = {}
        for z in Z3_ANGLES:
            idx = np.argmin(np.abs((edges[:-1] + edges[1:]) / 2 - z))
            z_dens[f"{np.degrees(z):.0f}deg"] = h[idx]
        print(f"    Z3 densities: {z_dens} (bg median={np.median(h):.4f})")

    # --- 4. combined Kekule phase: triple product of three inequivalent
    #        single-peak components (q sum ~ 0, fast oscillations cancel) ---
    combo = triple_product_combo(q_by_name)
    q_sum = np.sum([q_by_name[c] for c in combo], axis=0)
    print(f"combined triple: {combo}, q_sum = ({q_sum[0]:.3f}, {q_sum[1]:.3f}) px")
    P = None
    for cname in combo:
        P = psi_by_name[cname] if P is None else P * psi_by_name[cname]
    theta = combined_theta(P, q_sum, topo)
    ampP = np.abs(P)
    threshP = np.percentile(ampP[valid], 50)
    goodP = valid & (ampP > threshP)
    plot_combined(theta, goodP, ampP, combo,
                  outdir / "combined_kekule_phi.png")
    np.save(outdir / "combined_theta_field.npy", theta)
    np.save(outdir / "combined_product_psi.npy", P)

    # --- summary figures: 2x3 phi histograms and 2x3 phi maps ---
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (name, (h, edges)) in zip(axs.ravel(), histograms.items(), strict=False):
        ax.plot((edges[:-1] + edges[1:]) / 2, h, color="steelblue", lw=1.2)
        for z in Z3_ANGLES:
            ax.axvline(z, color="red", ls="--", lw=1.0)
        ax.set_xlabel("phi (rad)", fontsize=11)
        ax.set_ylabel("density", fontsize=11)
        ax.set_title(f"{name} Kekule phi", fontsize=12)
        ax.set_xticks([0, np.pi / 3, 2 * np.pi / 3, np.pi,
                       4 * np.pi / 3, 5 * np.pi / 3, 2 * np.pi])
        ax.set_xticklabels(["0", "", "2pi/3", "", "4pi/3", "", "2pi"], fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "r3_peaks_kekule_phi_hist.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (name, phi_field) in zip(axs.ravel(), phi_fields.items(), strict=False):
        im = ax.imshow(np.where(valid, phi_field, np.nan), cmap="hsv",
                       origin="lower", vmin=0, vmax=2 * np.pi)
        ax.set_title(f"{name} phi(r)", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axs, fraction=0.02, pad=0.02)
    fig.tight_layout()
    fig.savefig(outdir / "r3_peaks_kekule_phi_map.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    # --- 0/pi histograms (row 1: OUT + nomask, row 2: IN) ---
    bins = np.linspace(-np.pi, np.pi, args.bins + 1)
    first_name = next(iter(phi_fields))
    row1 = [
        ("nomask (FFT2)", hist_phase(fft2, valid, use_valid=False)),
        ("r3all mask OUT", hist_phase(c_out, valid, use_valid=True)),
        (f"{first_name} mask OUT",
         hist_phase(np.load(outdir / f"{first_name}_mask_out_ifft.npy"),
                    valid, use_valid=True)),
    ]
    row2 = [
        None,
        ("r3all mask IN", hist_phase(c_in, valid, use_valid=True)),
        (f"{first_name} mask IN",
         hist_phase(np.load(outdir / f"{first_name}_mask_in_ifft.npy"),
                    valid, use_valid=True)),
    ]
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    for row, panels in enumerate([row1, row2]):
        for col, item in enumerate(panels):
            ax = axs[row, col]
            if item is None:
                ax.axis("off")
                continue
            title, p = item
            ax.hist(p, bins=bins, color="steelblue", edgecolor="none")
            ax.set_xlabel("Phase (rad)", fontsize=13)
            ax.set_ylabel("Count", fontsize=13)
            ax.set_title(f"{title} - Phase ({args.bins} bins)", fontsize=13)
            ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
            ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0",
                                r"$\pi/2$", r"$\pi$"], fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir / "all_cases_phase_histograms.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    print("done")


if __name__ == "__main__":
    main()
