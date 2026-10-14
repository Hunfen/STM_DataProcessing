# ruff: noqa
# fmt: off
# Byte-for-byte archive of the historical one-off lock-in script: kept as a record, never imported and not part of the runtime, so the body is frozen and both ruff gates are waived for this file.
"""Preview of the r3 (sqrt3 x sqrt3 R30) q-space mask on the corrected map. v2

Fixes relative to v1: the overview no longer draws unlabelled zoom boxes (they
were mistaken for the mask); the mask is shown at its true scale in a dedicated
panel; the pointwise zooms are replaced by radial line cuts, which is what
actually shows what the window passes.

The mask drawn here is the window used by skills/local-q-map:
    psi_q(r) = IFFT{ FFT[ T(r) exp(-i q.r) ] * exp(-lambda^2 |k|^2 / 2) }
i.e. an isotropic Gaussian disk of width sigma_q = 1/lambda centred on the peak.

Run:
  MPLCONFIGDIR=<writable> PYTHONDONTWRITEBYTECODE=1 \
    <repo>/.venv/bin/python make_r3_mask_preview.py
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from scipy.ndimage import map_coordinates

HERE = Path(__file__).resolve().parent
DATA = HERE.parent
STEM = "20251112_topo0009_1mV100pA_100nm"
H5 = DATA / f"{STEM}_corrected.h5"

LAMBDA_NM = 13.0
A_NM = 0.246
PAIRS = [
    {"g": "g1", "r_px": 280.44, "az_deg": 29.95, "amp": 1.188e-06},
    {"g": "g2", "r_px": 280.00, "az_deg": 89.80, "amp": 1.111e-06},
    {"g": "g3", "r_px": 280.94, "az_deg": 149.88, "amp": 5.503e-07},
]
LEVELS = (1.0, 2.0, 3.0)
CUT_HALF_PX = 12.0
CUT_STEP = 0.05


def centre_xy(n_px: int, r_px: float, az_deg: float) -> tuple[float, float]:
    c = n_px // 2
    th = np.radians(az_deg)
    return c + r_px * np.cos(th), c + r_px * np.sin(th)


def circle_outline(cx: float, cy: float, rho: float, n: int = 240) -> tuple[np.ndarray, np.ndarray]:
    phi = np.linspace(0.0, 2.0 * np.pi, n)
    return cx + rho * np.cos(phi), cy + rho * np.sin(phi)


def badge(ax, letter: str) -> None:
    """Big high-contrast panel letter, drawn inside the axes."""
    ax.text(0.012, 0.988, f"({letter})", transform=ax.transAxes, fontsize=30, fontweight="bold",
            va="top", ha="left", zorder=60, color="black",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="black", alpha=0.9))


def main() -> None:
    with h5py.File(H5, "r") as f:
        arr = f["corrected"][:]
        fov_nm = float(f.attrs["field_of_view_nm"])
        nm_per_px = float(f.attrs["nm_per_px"])
    n = arr.shape[0]
    c = n // 2
    lam_px = LAMBDA_NM / nm_per_px
    sigma_q = n / (2.0 * np.pi * lam_px)
    fwhm_q = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma_q

    log_amp = np.log10(np.maximum(np.abs(np.fft.fftshift(np.fft.fft2(np.nan_to_num(arr, nan=0.0)))), 1e-30))

    fig = plt.figure(figsize=(20.5, 12.0), dpi=150)
    gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.24)

    # ---- (a) polar annulus view --------------------------------------------
    rad = np.arange(250.0, 312.0, 0.25)
    az = np.arange(0.0, 360.0, 0.25)
    az_g, rad_g = np.meshgrid(az, rad)
    thg = np.radians(az_g)
    polar = map_coordinates(log_amp, [c + rad_g * np.sin(thg), c + rad_g * np.cos(thg)],
                            order=1, mode="nearest")
    vmax_a = float(polar.max())
    ax = fig.add_subplot(gs[0, 0])
    badge(ax, "a")
    im = ax.imshow(polar, origin="lower", aspect="auto", cmap="plasma",
                   extent=(0.0, 360.0, rad[0], rad[-1]), vmin=vmax_a - 3.0, vmax=vmax_a)
    for p in PAIRS:
        for sign in (1.0, -1.0):
            cx, cy = centre_xy(n, p["r_px"], p["az_deg"])
            if sign < 0:
                cx, cy = 2 * c - cx, 2 * c - cy
            for k, s in enumerate(LEVELS):
                ox, oy = circle_outline(cx, cy, s * sigma_q)
                rr = np.hypot(ox - c, oy - c)
                aa = np.degrees(np.arctan2(oy - c, ox - c)) % 360.0
                ax.plot(aa, rr, color="#00e5ff", lw=1.5 if k == 0 else 0.9,
                        alpha=0.95 if k == 0 else 0.55)
            rr0 = float(np.hypot(cx - c, cy - c))
            aa0 = float(np.degrees(np.arctan2(cy - c, cx - c)) % 360.0)
            ax.plot([aa0], [rr0], "o", ms=3.0, color="#00e5ff")
            ax.annotate(p["g"] if sign > 0 else "-" + p["g"], (aa0, rr0 + 4.2),
                        color="#00e5ff", ha="center", fontsize=9)
    ax.set_xlim(0, 360)
    ax.set_ylim(rad[0], rad[-1])
    ax.set_xlabel("azimuth in the FFT (deg)")
    ax.set_ylabel("radius in the FFT (px)")
    ax.set_title("(a) r3 annulus, log10|FFT| (3 decades)\n"
                 "cyan rings = the mask at 1/2/3 sigma", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.036, pad=0.02)

    # ---- (b) full FFT overview ---------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    badge(ax, "b")
    step = 4
    ax.imshow(log_amp[::step, ::step], origin="lower", cmap="plasma",
              vmin=vmax_a - 3.0, vmax=vmax_a)
    r1 = 2.0 * fov_nm / (np.sqrt(3.0) * A_NM)
    for radius, colour, lab in ((r1, "#00ff7f", "1x1 ring"), (r1 / np.sqrt(3.0), "#00e5ff", "r3 ring")):
        ax.add_patch(Circle((n / 2 / step, n / 2 / step), radius / step, fill=False,
                            color=colour, lw=1.0, alpha=0.9))
        ax.annotate(lab, (n / 2 / step + radius / step + 2, n / 2 / step), color=colour, fontsize=9)
    for p in PAIRS:
        cx, cy = centre_xy(n, p["r_px"], p["az_deg"])
        ax.plot([cx / step], [cy / step], "o", ms=4, mfc="none", mec="#00e5ff", mew=1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("(b) full |FFT| (4x decimated, same colour window)\n"
                 "open circles = the three r3 spots; the mask (3 px) is far too\n"
                 "small to be drawn at this scale -- see panel (c)", fontsize=11)

    # ---- (c) the mask itself, true scale -----------------------------------
    ax = fig.add_subplot(gs[0, 2])
    badge(ax, "c")
    half = 8
    gx = np.arange(-half, half + 1, 1.0)
    X, Y = np.meshgrid(gx, gx)
    mask = np.exp(-(X ** 2 + Y ** 2) / (2.0 * sigma_q ** 2))
    ax.imshow(mask, origin="lower", cmap="GnBu", extent=(gx[0] - .5, gx[-1] + .5, gx[0] - .5, gx[-1] + .5))
    for s in LEVELS:
        ax.add_patch(Circle((0, 0), s * sigma_q, fill=False, color="crimson", lw=1.2))
    ax.plot([0], [0], "+", color="crimson", ms=8)
    for s, lab in zip(LEVELS, ("1σ", "2σ", "3σ")):
        ax.annotate(lab, (s * sigma_q + 0.15, -0.15), color="crimson", fontsize=9)
    for d, lab in ((2, "2px"), (3, "3px"), (5, "5px")):
        ax.plot([d], [0], "v", color="k", ms=4)
        ax.annotate(lab, (d, 0.35), fontsize=8, ha="center")
    ax.set_xlabel("FFT px relative to the peak centre")
    ax.set_ylabel("FFT px")
    ax.set_title("(c) THE MASK ITSELF, true scale (17x17 px shown)\n"
                 f"sigma_q = {sigma_q:.3f} px, FWHM = {fwhm_q:.2f} px,  3σ = {3 * sigma_q:.2f} px\n"
                 f"weight: 2px {np.exp(-4 / (2 * sigma_q ** 2)):.2f}, "
                 f"3px {np.exp(-9 / (2 * sigma_q ** 2)):.2f}, "
                 f"5px {np.exp(-25 / (2 * sigma_q ** 2)):.1e}", fontsize=11)
    fig.colorbar(plt.cm.ScalarMappable(cmap="GnBu", norm=plt.Normalize(0, 1)), ax=ax,
                 fraction=0.036, pad=0.02, label="mask weight")

    # ---- (d) radial cuts through the three peaks ---------------------------
    ax = fig.add_subplot(gs[1, 0])
    badge(ax, "d")
    ts = np.arange(-CUT_HALF_PX, CUT_HALF_PX + CUT_STEP, CUT_STEP)
    colours = ("#d62728", "#1f77b4", "#2ca02c")
    for p, col in zip(PAIRS, colours):
        cx, cy = centre_xy(n, p["r_px"], p["az_deg"])
        th = np.radians(p["az_deg"])
        rows = cy + ts * np.sin(th)
        cols = cx + ts * np.cos(th)
        cut = map_coordinates(log_amp, [rows, cols], order=1, mode="nearest")
        ax.plot(ts, cut, color=col, lw=1.4, label=f"{p['g']} ({p['az_deg']:.1f}°)")
    ax.axvline(0.0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel("offset from the peak centre along the radial direction (px)")
    ax.set_ylabel("log10 |FFT|")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("(d) radial cuts through g1/g2/g3\n"
                 "peak FWHM 1.9-3.4 px; the small bumps at |t| > 10 px are the\n"
                 "radial sidelobes of g2 (the 'other structures' near r3)", fontsize=11)

    # ---- (e) zoom on g2 with a LOCAL colour window --------------------------
    ax = fig.add_subplot(gs[1, 1])
    badge(ax, "e")
    p = PAIRS[1]
    cx, cy = centre_xy(n, p["r_px"], p["az_deg"])
    s = 12
    x0, y0 = int(round(cx)), int(round(cy))
    sub = log_amp[y0 - s:y0 + s + 1, x0 - s:x0 + s + 1]
    ax.imshow(sub, origin="lower", cmap="plasma", interpolation="nearest",
              extent=(x0 - s - .5, x0 + s + .5, y0 - s - .5, y0 + s + .5),
              vmin=float(sub.max()) - 2.0, vmax=float(sub.max()))
    for k, sig in enumerate(LEVELS):
        ax.add_patch(Circle((cx, cy), sig * sigma_q, fill=False, color="#00e5ff",
                            lw=1.6 if k == 0 else 1.0, alpha=0.95 if k == 0 else 0.55))
    ax.set_xlabel("FFT x (px)")
    ax.set_ylabel("FFT y (px)")
    ax.set_title("(e) g2 zoom, 25x25 px, LOCAL colour window (2 decades)\n"
                 "cyan rings = 1/2/3 sigma of the mask; the mask encloses the core\n"
                 "and nothing else", fontsize=11)

    # ---- (f) numbers --------------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    badge(ax, "f")
    ax.axis("off")
    lines = [
        f"input: {STEM}_corrected.h5",
        f"canvas {n} x {n} px, FOV {fov_nm:.4f} nm, {nm_per_px:.6f} nm/px",
        f"NaN -> 0 (same rule as skills/local-q-map)",
        "",
        f"lambda = {LAMBDA_NM:.0f} nm  ->  sigma_q = {sigma_q:.4f} px",
        f"mask FWHM = {fwhm_q:.3f} px,  3 sigma = {3 * sigma_q:.3f} px",
        "",
        "mask weight at radial offset:",
    ]
    for d in (2, 3, 4, 5, 6, 8, 10, 12):
        lines.append(f"    {d:>2} px : {np.exp(-(d ** 2) / (2 * sigma_q ** 2)):.3e}")
    lines += [
        "",
        "six r3 spots (measured on this map):",
        "    g1  30.0 deg, r = 280.44 px, amp 1.188e-06 (1.000)",
        "    g2  89.8 deg, r = 280.00 px, amp 1.111e-06 (0.935)",
        "    g3 149.9 deg, r = 280.94 px, amp 5.503e-07 (0.463)",
        "    plus the Friedel partners at +180 deg",
        "",
        "strongest non-peak structure in the annulus:",
        "    286.5 px, 29.25 deg -> 5.6 % of g1, 6.1 px from g1",
        "    -> the mask suppresses it to < 1e-3",
        "    g2 radial sidelobes at 259-304 px (3-4 %) are >= 10 px away",
        "    -> suppressed to ~3e-14",
    ]
    ax.text(0.0, 0.98, "\n".join(lines), va="top", ha="left", fontsize=10.5, family="monospace")

    fig.suptitle(
        f"r3 mask preview v2  |  lambda = {LAMBDA_NM:.0f} nm, sigma_q = {sigma_q:.3f} px "
        f"(mask FWHM {fwhm_q:.2f} px)  |  per-pair centres refined from THIS map",
        fontsize=12)
    out_png = HERE / "r3_mask_preview.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "input_h5": str(H5),
        "canvas_px": n,
        "field_of_view_nm": fov_nm,
        "nm_per_px": nm_per_px,
        "nan_fill_rule": "NaN -> 0 (same as skills/local-q-map)",
        "lambda_nm": LAMBDA_NM,
        "lambda_px": lam_px,
        "sigma_q_px": sigma_q,
        "mask_fwhm_px": fwhm_q,
        "mask_weight_at_px": {str(d): float(np.exp(-(d ** 2) / (2 * sigma_q ** 2)))
                              for d in (1, 2, 3, 4, 5, 6, 8, 10, 12)},
        "pairs": [dict(p, amp_relative_to_strongest=p["amp"] / PAIRS[0]["amp"]) for p in PAIRS],
        "nearest_non_peak_structure": {"r_px": 286.53, "az_deg": 29.25,
                                       "amp_fraction_of_g1": 0.0561, "distance_to_g1_px": 6.1},
        "note": "centres measured on this map (5-px non-maximum suppression, 255-305 px "
                "annulus); the LF report reference lives in the INPUT frame and is not used",
    }
    (HERE / "r3_mask_preview.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out_png}; sigma_q = {sigma_q:.4f} px, FWHM = {fwhm_q:.4f} px")


if __name__ == "__main__":
    main()
