# ruff: noqa
# fmt: off
# Byte-for-byte archive of the historical one-off lock-in script: kept as a record, never imported and not part of the runtime, so the body is frozen and both ruff gates are waived for this file.
"""Presentation figure: how the spatial lock-in is done on the r3 peaks.

One row, five panels, one short caption each:
  1 FFT of the corrected topography      -> where the r3 ring is
  2 zoom on one r3 spot + the mask       -> what is kept (±g)
  3 B(r) back in real space (full field) -> the √3 component
  4 demodulated amplitude |psi|
  5 demodulated phase arg psi

Output: lockin_steps_v3.png (new file name; the diagnostic plots are left alone)
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

HERE = Path(__file__).resolve().parent
DATA = HERE.parent
STEM = "20251112_topo0009_1mV100pA_100nm"
REPORT = HERE / "local_q_map_report.json"

Q_INDEX = 0          # demo pair: g1 (~30 deg) and its Friedel partner
FFT_ZOOM = 8         # px half-width of the q-space zoom (the mask is ~3 px wide)
PHASE_WIN_DEG = 20.0


def main() -> None:
    rep = json.loads(REPORT.read_text(encoding="utf-8"))
    n = int(rep["canvas_px"])
    nm_per_px = float(rep["nm_per_px"])
    lam_nm = float(rep["lambda_nm"])
    sigma_q = n / (2.0 * np.pi * (lam_nm / nm_per_px))
    q = rep["q"][Q_INDEX]
    qx, qy = (float(v) for v in q["q_rad_px"])
    qpx = np.asarray(q["q_px"], dtype=float)
    az = float(np.degrees(np.arctan2(qy, qx)) % 360.0)

    rows = np.arange(n)[:, None].astype(float)
    cols = np.arange(n)[None, :].astype(float)
    with h5py.File(DATA / f"{STEM}_corrected.h5", "r") as f:
        arr = f["corrected"][:]
    valid = np.isfinite(arr)
    fftamp = np.abs(np.fft.fftshift(np.fft.fft2(np.nan_to_num(arr, nan=0.0))))
    logamp = np.log10(np.maximum(fftamp, 1e-30))

    with h5py.File(q["artifacts"]["h5"], "r") as f:
        field = f["field"][:]
        amp = f["amplitude"][:]
        theta = np.degrees(np.angle(np.exp(1j * f["theta"][:])))
        mask = f["mask"][:]
    B = 2.0 * np.real(field * np.exp(1j * (qx * cols + qy * rows)))
    good = mask > 0.5
    mean_deg = float(np.degrees(np.angle(np.mean(np.exp(1j * np.radians(theta[good]))))))
    dtheta = np.degrees(np.angle(np.exp(1j * np.radians(theta - mean_deg))))

    c = n // 2
    gx, gy = int(round(c + qpx[0])), int(round(c + qpx[1]))
    field_nm = n * nm_per_px
    ext = (0.0, field_nm, 0.0, field_nm)
    Bmap = np.where(valid, B, np.nan)
    Amap = np.where(valid, amp, np.nan)

    fig = plt.figure(figsize=(21.0, 5.4), dpi=150)
    gs = fig.add_gridspec(1, 5, wspace=0.22)

    # ---- 1  FFT overview ------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    step = 4
    ax.imshow(logamp[::step, ::step], origin="lower", cmap="plasma",
              vmin=float(logamp.max()) - 3.0, vmax=float(logamp.max()))
    r1 = 2.0 * float(rep["field_of_view_nm"]) / (np.sqrt(3.0) * 0.246)
    for radius, colour in ((r1, "#00ff7f"), (r1 / np.sqrt(3.0), "#00e5ff")):
        ax.add_patch(Circle((c / step, c / step), radius / step, fill=False, color=colour, lw=1.8))
    for sx in (+1, -1):
        ax.add_patch(Circle(((c + sx * qpx[0]) / step, (c + sx * qpx[1]) / step),
                            16.0 / step, fill=False, color="w", lw=2.0))
    ax.annotate("1×1", (c / step + r1 / step + 2, c / step - 3), color="#00ff7f", fontsize=12)
    ax.annotate("r3", (c / step + (r1 / np.sqrt(3.0)) / step + 2, c / step - 3),
                color="#00e5ff", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("1. FFT of the topography", fontsize=16)

    # ---- 2  mask --------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    z = logamp[gy - FFT_ZOOM:gy + FFT_ZOOM + 1, gx - FFT_ZOOM:gx + FFT_ZOOM + 1]
    ax.imshow(z, origin="lower", cmap="plasma", interpolation="nearest",
              extent=(-FFT_ZOOM, FFT_ZOOM, -FFT_ZOOM, FFT_ZOOM),
              vmin=float(z.max()) - 2.0, vmax=float(z.max()))
    for k, s in enumerate((1.0, 2.0, 3.0)):
        ax.add_patch(Circle((0, 0), s * sigma_q, fill=False, color="#00e5ff",
                            lw=2.2 if k == 0 else 1.2, alpha=1.0 if k == 0 else 0.7))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("2. keep ±g only", fontsize=16)
    ax.set_xlabel("Gaussian, FWHM 3 px", fontsize=12)

    # ---- 3..5  real space -----------------------------------------------------
    specs = (
        (Bmap, "3. √3 component B(r)", "viridis", float(np.nanpercentile(np.abs(Bmap), 99.0))),
        (Amap, "4. demodulated |ψ|", "inferno", None),
        (np.where(valid, dtheta, np.nan), "5. demodulated θ", "twilight", PHASE_WIN_DEG),
    )
    tick_nm = [0, 25, 50, 75, 100]
    for i, (img, title, cmap, win) in enumerate(specs):
        ax = fig.add_subplot(gs[0, 2 + i])
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad("0.75")
        if cmap == "inferno":
            kw = dict(vmin=float(np.nanpercentile(img, 2.0)),
                      vmax=float(np.nanpercentile(img, 99.5)))
        elif cmap == "twilight":
            kw = dict(vmin=-win, vmax=win)
        else:
            kw = dict(vmin=-win, vmax=win)
        im = ax.imshow(img, origin="lower", cmap=cm, extent=ext, **kw)
        ax.set_xticks(tick_nm); ax.set_yticks(tick_nm)
        ax.tick_params(labelsize=10)
        ax.set_xlabel("nm", fontsize=12)
        ax.set_title(title, fontsize=16)
        if cmap == "twilight":
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=9)

    fig.suptitle(
        "spatial lock-in on one r3 pair:   "
        r"$\psi(r)=\mathrm{LPF}\,[\,T(r)\,e^{-i g\cdot r}\,]$"
        f"      g at {az:.0f}°, |g| = {q['q_nm_inv_abs']:.1f} nm⁻¹, λ = {lam_nm:.0f} nm",
        fontsize=17)
    out = HERE / "lockin_steps_v3.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
