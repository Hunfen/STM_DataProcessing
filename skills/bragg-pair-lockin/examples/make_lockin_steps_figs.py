# ruff: noqa
# fmt: off
# Byte-for-byte archive of the historical one-off lock-in script: kept as a record, never imported and not part of the runtime, so the body is frozen and both ruff gates are waived for this file.
"""Four separate step figures for the r3 spatial lock-in.

step1_fft_pairs.png        FFT of the corrected topography, cropped exactly like
                           <stem>_corrected_fft_30nm304px_plasma.png (central
                           1051 x 1051 px = +-5.07 cycles/nm), with the three r3
                           pairs (g_i, -g_i) marked in three colours and the 1x1
                           and r3 rings drawn.
step2_q_zoom_one_r3.png    +-8 q-px zoom on one r3 spot with the Gaussian low-pass
                           circles (1 sigma, 2 sigma, 3 sigma); the spot position
                           is taken inside the same crop as step 1.
step3_modulated_field.png  modulus and argument of the real-space modulated field
                           X(r) = T(r) exp(-i g1.r)  (before any low-pass).
step4_lowpassed_field.png  modulus and argument of the low-passed field
                           psi(r) = LPF[T exp(-i g1.r)]  (the local-q-map product).

Inputs (read-only): ../local_q_map_report.json, the per-q h5 products it names
(basenames only: the report's absolute paths predate the folder rename), and
../<STEM>_corrected.h5 (corrected topography and its official fft2).

All labels are matplotlib mathtext (mathtext.fontset = dejavusans, no usetex).
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

HERE = Path(__file__).resolve().parent
R3 = HERE.parent  # .../r3_lockin
DATA = R3.parent  # dataset directory
STEM = "20251112_topo0009_1mV100pA_100nm"
REPORT = R3 / "local_q_map_report.json"

Q_INDEX = 0  # demo pair: g1 = q0 = (1/3, 1/3), azimuth 29.9 deg
HALF = 525  # crop half width in px (same crop as the _30nm304px_plasma figure)
FFT_ZOOM = 8  # q-space half width of the step-2 zoom, in q-px
PAIR_COLOURS = ("#00ff7f", "#00e5ff", "#ff4fd8")  # one colour per r3 pair
PHASE_WIN_DEG = 20.0  # display window of the step-4 phase
DPI = 220
BAD_COLOUR = "0.55"
STROKE = [path_effects.withStroke(linewidth=2.5, foreground="black")]


def sci(value, digits=2):
    """Matplotlib-math scientific notation, e.g. $2.85\\times10^{-11}$."""
    exponent = int(np.floor(np.log10(abs(float(value)))))
    mantissa = float(value) / 10.0**exponent
    return rf"${mantissa:.{digits}f}\times10^{{{exponent}}}$"


def load():
    rep = json.loads(REPORT.read_text(encoding="utf-8"))
    n = int(rep["canvas_px"])
    nm_per_px = float(rep["nm_per_px"])
    with h5py.File(DATA / f"{STEM}_corrected.h5", "r") as f:
        arr = f["corrected"][:]
        fft2 = f["fft2"][:]
    return rep, n, nm_per_px, arr, fft2


def q_h5(q):
    """Per-q product path: the report's absolute path predates the rename."""
    return R3 / Path(q["artifacts"]["h5"]).name


def topo_cmap():
    """The package's gwyddion topography colormap (the one the LF previews use)."""
    from stm_data_processing.stm.preview_plot import gwyddion

    return gwyddion.copy()


def with_bad(cmap, colour=BAD_COLOUR):
    cmap = cmap.copy()
    cmap.set_bad(colour)
    return cmap


def figure_1_fft(rep, n, fft2, out_dir):
    mag = np.abs(fft2)
    c = n // 2
    crop = mag[c - HALF : c + HALF + 1, c - HALF : c + HALF + 1]
    lo, hi = (float(v) for v in np.percentile(mag, [5.0, 99.5]))
    plane = np.clip(
        (np.log1p(crop) - np.log1p(lo)) / (np.log1p(hi) - np.log1p(lo)), 0.0, 1.0
    )
    r1 = float(np.hypot(*rep["basis_source"]["b1_px"]))  # 1x1 ring radius, px
    r3 = r1 / np.sqrt(3.0)

    fig, ax = plt.subplots(figsize=(7.8, 7.8))
    ax.imshow(plane, cmap="plasma", origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    for radius, style, label in (
        (r1, (0, (7, 5)), r"$1\times1$"),
        (r3, (0, (3, 3)), r"$r_3$"),
    ):
        ax.add_patch(
            Circle((HALF, HALF), radius, fill=False, lw=1.4, ls=style, color="white", alpha=0.9)
        )
        ax.text(
            HALF + radius + 5, HALF - 8, label, color="white", fontsize=13,
            va="center", path_effects=STROKE,
        )
    for i, q in enumerate(rep["q"]):
        qx, qy = (float(v) for v in q["q_px"])
        colour = PAIR_COLOURS[i % len(PAIR_COLOURS)]
        for s in (+1.0, -1.0):
            ax.add_patch(
                Circle((HALF + s * qx, HALF + s * qy), 16.0, fill=False, lw=2.4, color=colour)
            )
        ax.text(
            HALF + qx + 20, HALF + qy + 20, str(i + 1), color=colour, fontsize=15,
            fontweight="bold", path_effects=STROKE,
        )
        print(
            f"  pair {i + 1}: (h,k)=({q['h']:.3f},{q['k']:.3f})  "
            f"|q| = {q['q_nm_inv_abs']:.4f} rad/nm = {q['q_nm_inv_abs'] / (2 * np.pi):.4f} cycles/nm"
        )
    ax.set_title(
        r"1  FFT of the corrected topography (cropped, $\pm5.07\ \mathrm{cycles/nm}$)" "\n"
        r"three colour-coded $r_3$ pairs $(g_i,\,-g_i)$;  dashed: $1\times1$ and $r_3$ rings",
        fontsize=12,
    )
    out = out_dir / "step1_fft_pairs.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")

    # circles really sit on ring members?
    print("  |F| at the six marked spots vs the overall median |F|:")
    for i, q in enumerate(rep["q"]):
        qx, qy = (int(round(v)) for v in q["q_px"])
        vals = [mag[c + s * qy, c + s * qx] for s in (+1, -1)]
        print(f"    pair {i + 1}: {vals[0]:.3e} / {vals[1]:.3e}")
    print(f"    median |F| over the crop: {np.median(crop):.3e}")


def figure_2_zoom(rep, n, fft2, out_dir):
    q = rep["q"][Q_INDEX]
    lam_nm = float(rep["lambda_nm"])
    nm_per_px = float(rep["nm_per_px"])
    sigma_q = n / (2.0 * np.pi * (lam_nm / nm_per_px))
    azimuth = np.degrees(np.arctan2(q["q_rad_px"][1], q["q_rad_px"][0])) % 360.0
    # same crop as step 1, then locate the spot INSIDE the crop:
    # the spot sits at (HALF + q_px[0], HALF + q_px[1]) of the cropped frame.
    c = n // 2
    crop = np.abs(fft2)[c - HALF : c + HALF + 1, c - HALF : c + HALF + 1]
    logcrop = np.log10(np.maximum(crop, 1e-30))
    gx = int(round(HALF + float(q["q_px"][0])))
    gy = int(round(HALF + float(q["q_px"][1])))
    z = logcrop[gy - FFT_ZOOM : gy + FFT_ZOOM + 1, gx - FFT_ZOOM : gx + FFT_ZOOM + 1]
    iy, ix = np.unravel_index(np.argmax(z), z.shape)
    print(
        f"  spot window: crop centre ({gx}, {gy}), peak offset from the window centre "
        f"= ({ix - FFT_ZOOM}, {iy - FFT_ZOOM}) q-px; peak |F| = {10 ** z.max():.3e}, "
        f"crop median |F| = {np.median(crop):.3e}"
    )

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.imshow(
        z, origin="lower", cmap="plasma", interpolation="nearest",
        extent=(-FFT_ZOOM, FFT_ZOOM, -FFT_ZOOM, FFT_ZOOM),
        vmin=float(z.max()) - 2.0, vmax=float(z.max()),
    )
    for k, s in enumerate((1.0, 2.0, 3.0)):
        ax.add_patch(
            Circle((0.0, 0.0), s * sigma_q, fill=False, color="#00e5ff",
                   lw=2.4 if k == 0 else 1.4, alpha=1.0 if k == 0 else 0.7)
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r"$q-g_1$   (q-px)", fontsize=11)
    ax.set_title(
        rf"2  zoom on one $r_3$ spot:  $g_1=(1/3,\,1/3)$,  azimuth ${azimuth:.1f}^\circ$" "\n"
        rf"the low-pass keeps $\pm g_1$ only;  circles $1\sigma_q,\,2\sigma_q,\,3\sigma_q$  "
        rf"($\sigma_q={sigma_q:.3f}$ q-px $=1/2\pi\lambda$, FWHM $={2.3548 * sigma_q:.2f}$ q-px)",
        fontsize=11,
    )
    out = out_dir / "step2_q_zoom_one_r3.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def figure_3_modulated(rep, n, nm_per_px, arr, out_dir):
    q = rep["q"][Q_INDEX]
    qx, qy = (float(v) for v in q["q_rad_px"])
    rows = np.arange(n)[:, None].astype(float)
    cols = np.arange(n)[None, :].astype(float)
    field = arr * np.exp(-1j * (qx * cols + qy * rows))
    amp = np.abs(field)
    ang = np.degrees(np.angle(field))
    finite = np.isfinite(amp)
    vmax = float(np.percentile(amp[finite], 99.9))
    period_px = 2.0 * np.pi / float(np.hypot(qx, qy))
    fov = n * nm_per_px
    ext = (0.0, fov, 0.0, fov)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.6))
    im0 = axes[0].imshow(amp, origin="lower", cmap=with_bad(topo_cmap()), extent=ext,
                         vmin=0.0, vmax=vmax)
    im1 = axes[1].imshow(ang, origin="lower", cmap=with_bad(plt.get_cmap("twilight")), extent=ext,
                         vmin=-180.0, vmax=180.0)
    for ax in axes:
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xlabel("nm", fontsize=11)
    axes[0].set_title(
        rf"$|X|$   (m)    window $[0,\,p_{{99.9}}={sci(vmax)}]$", fontsize=12
    )
    axes[1].set_title(
        r"$\arg X$   ($^\circ$)    window $[-180^\circ,\,180^\circ]$", fontsize=12
    )
    cb = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=9)
    fig.suptitle(
        r"3  real-space modulation  $X(r)=T(r)\,e^{-i g_1\cdot r}$   (BEFORE the low-pass)" "\n"
        rf"the modulation changes only the phase: $|X|=|T|$;  the carrier winds $2\pi$ every "
        rf"${period_px:.2f}$ px $={period_px * nm_per_px:.3f}$ nm;  "
        r"$\arg X$ jumps by $\pi$ wherever $T<0$",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = out_dir / "step3_modulated_field.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}   (|X| p99.9 = {vmax:.3e} m; carrier period {period_px:.2f} px)")


def figure_4_lowpassed(rep, n, nm_per_px, out_dir):
    q = rep["q"][Q_INDEX]
    with h5py.File(q_h5(q), "r") as f:
        amp = f["amplitude"][:]
        theta = f["theta"][:]
        mask = f["mask"][:]
    good = mask > 0.5
    mean_deg = float(np.degrees(np.angle(np.mean(np.exp(1j * theta[good]))))) % 360.0
    dtheta = np.degrees(np.angle(np.exp(1j * theta) * np.exp(-1j * np.radians(mean_deg))))
    shown_amp = np.where(good, amp, np.nan)
    shown_ang = np.where(good, dtheta, np.nan)
    vmin_a = float(np.nanpercentile(shown_amp, 2.0))
    vmax_a = float(np.nanpercentile(shown_amp, 99.5))
    fov = n * nm_per_px
    ext = (0.0, fov, 0.0, fov)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.6))
    im0 = axes[0].imshow(shown_amp, origin="lower", cmap=with_bad(plt.get_cmap("inferno")),
                         extent=ext, vmin=vmin_a, vmax=vmax_a)
    im1 = axes[1].imshow(shown_ang, origin="lower", cmap=with_bad(plt.get_cmap("twilight")),
                         extent=ext, vmin=-PHASE_WIN_DEG, vmax=PHASE_WIN_DEG)
    for ax in axes:
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xlabel("nm", fontsize=11)
    axes[0].set_title(
        rf"$|\psi|$   (m)    window $p_2$–$p_{{99.5}}$:  {sci(vmin_a)} … {sci(vmax_a)}",
        fontsize=12,
    )
    axes[1].set_title(
        rf"$\arg\psi-\langle\arg\psi\rangle$   ($^\circ$)    "
        rf"window $\pm{PHASE_WIN_DEG:.0f}^\circ$",
        fontsize=12,
    )
    cb = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=9)
    fig.suptitle(
        r"4  after the Gaussian low-pass ($\lambda=13$ nm):  "
        r"$\psi(r)=\mathrm{LPF}\,[\,T(r)\,e^{-i g_1\cdot r}\,]$" "\n"
        rf"local amplitude $|\psi|=A_{{g_1}}/2$ and local phase $\arg\psi=\varphi_{{g_1}}$;  "
        rf"$\langle\arg\psi\rangle={mean_deg:.2f}^\circ$ over the mask "
        rf"({good.mean() * 100:.2f} % valid),  grey = invalid",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = out_dir / "step4_lowpassed_field.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}   (mean phase {mean_deg:.3f} deg, mask coverage {good.mean() * 100:.3f} %)")


def main() -> None:
    out_dir = HERE
    rep, n, nm_per_px, arr, fft2 = load()
    print(
        rf"canvas {n}^2, {nm_per_px:.8f} nm/px, lambda = {rep['lambda_nm']} nm, "
        f"{len(rep['q'])} q's"
    )
    figure_1_fft(rep, n, fft2, out_dir)
    figure_2_zoom(rep, n, fft2, out_dir)
    figure_3_modulated(rep, n, nm_per_px, arr, out_dir)
    figure_4_lowpassed(rep, n, nm_per_px, out_dir)


if __name__ == "__main__":
    main()
