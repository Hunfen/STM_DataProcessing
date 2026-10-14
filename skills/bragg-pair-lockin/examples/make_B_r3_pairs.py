# ruff: noqa
# fmt: off
# Byte-for-byte archive of the historical one-off lock-in script: kept as a record, never imported and not part of the runtime, so the body is frozen and both ruff gates are waived for this file.
"""Reconstruct the three r3 B(r) maps from the local-q-map products.

B_i(r) = 2 Re[ psi_{q_i}(r) exp(i q_i . r) ]      <- the q_i component of the map

The complex field psi_q comes from skills/local-q-map (window lambda = 13 nm).
An independent check is run next to it: the same component extracted directly by
masking +/-q with the equivalent Gaussian in the raw FFT and inverse transforming.

Outputs (into this folder):
  B_r3_pair{1,2,3}_bare.png     1:1 pixel, no axes/colorbar, viridis, NaN grey
  B_r3_three_pairs_v{N}.png     three panels with a shared absolute colour scale
  B_r3_pairs.json               q vectors, amplitudes, cross-check numbers
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

HERE = Path(__file__).resolve().parent
DATA = HERE.parent
STEM = "20251112_topo0009_1mV100pA_100nm"
H5 = DATA / f"{STEM}_corrected.h5"
REPORT = HERE / "local_q_map_report.json"
FIGURE_TAG = "v1"


def load_corrected() -> tuple[np.ndarray, float, float]:
    with h5py.File(H5, "r") as f:
        arr = f["corrected"][:]
        return arr, float(f.attrs["field_of_view_nm"]), float(f.attrs["nm_per_px"])


def grey_nan_cmap(name: str, colour: str = "0.55"):
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(colour)
    return cmap


def main() -> None:
    rep = json.loads(REPORT.read_text(encoding="utf-8"))
    n = int(rep["canvas_px"])
    nm_per_px = float(rep["nm_per_px"])
    lam_nm = float(rep["lambda_nm"])
    sigma_q_px = n / (2.0 * np.pi * (lam_nm / nm_per_px))  # same as the mask preview

    arr, fov_nm, _ = load_corrected()
    valid = np.isfinite(arr)
    x = np.nan_to_num(arr, nan=0.0)
    fft_raw = np.fft.fftshift(np.fft.fft2(x))

    rows = np.arange(n)[:, None].astype(np.float64)
    cols = np.arange(n)[None, :].astype(np.float64)

    entries = []
    for j, q in enumerate(rep["q"]):
        qx, qy = (float(v) for v in q["q_rad_px"])
        az = float(np.degrees(np.arctan2(qy, qx)) % 360.0)
        with h5py.File(q["artifacts"]["h5"], "r") as f:
            field = f["field"][:]
            mask_ok = f["mask"][:]
        carrier = np.exp(1j * (qx * cols + qy * rows))
        b_field = 2.0 * np.real(field * carrier)

        # --- independent direct extraction: Gaussian mask on +/-q in the raw FFT
        cx, cy = n // 2, n // 2
        qpx = np.asarray(q["q_px"], dtype=float)
        dx_p, dy_p = cols - (cx + qpx[0]), rows - (cy + qpx[1])
        dx_m, dy_m = cols - (cx - qpx[0]), rows - (cy - qpx[1])
        gauss = np.exp(-(dx_p ** 2 + dy_p ** 2) / (2.0 * sigma_q_px ** 2)) + \
                np.exp(-(dx_m ** 2 + dy_m ** 2) / (2.0 * sigma_q_px ** 2))
        b_direct = np.real(np.fft.ifft2(np.fft.ifftshift(fft_raw * gauss)))

        ok = mask_ok > 0.5
        b_use = np.where(valid, b_field, np.nan)
        diff = (b_field - b_direct)[valid]
        corr = float(np.corrcoef(b_field[valid], b_direct[valid])[0, 1])
        entries.append({
            "index": j,
            "label": q["label"],
            "h": q["h"], "k": q["k"],
            "q_px": q["q_px"], "q_nm_inv": q["q_nm_inv"],
            "q_nm_inv_abs": q["q_nm_inv_abs"],
            "azimuth_deg": az,
            "psi_median_abs": float(q["amplitude"]["median"]),
            "mask_coverage": float(q["mask_coverage_fraction"]),
            "B": b_use,
            "rms_nm": float(np.nanstd(b_use)),
            "max_abs_nm": float(np.nanmax(np.abs(b_use))),
            "valid_fraction": float(np.isfinite(b_use).mean()),
            "crosscheck_rms_ratio_direct_over_field": float(np.std(b_direct[valid]) / np.std(b_field[valid])),
            "crosscheck_corr_field_vs_direct": corr,
            "crosscheck_max_abs_diff": float(np.max(np.abs(diff))),
        })

    finite = np.concatenate([e["B"][np.isfinite(e["B"])] for e in entries])
    vmax = float(np.percentile(np.abs(finite), 99.5))
    vmin = -vmax

    order = sorted(range(len(entries)), key=lambda i: -entries[i]["psi_median_abs"])
    names = {}
    for rank, i in enumerate(order, start=1):
        names[entries[i]["label"]] = f"pair{rank}"
    for e in entries:
        e["pair_rank"] = names[e["label"]]

    cmap = grey_nan_cmap("viridis")
    for e in entries:
        fig = plt.figure(figsize=(n / 150.0, n / 150.0), dpi=150)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(e["B"], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_axis_off()
        out = HERE / f"B_r3_{e['pair_rank']}_{e['label']}_bare.png"
        fig.savefig(out, pad_inches=0)
        plt.close(fig)
        e["bare_png"] = str(out)

    fig = plt.figure(figsize=(19.0, 7.0), dpi=150)
    gs = fig.add_gridspec(1, 3, wspace=0.06)
    axes = []
    im = None
    for slot, i in enumerate(order):
        e = entries[i]
        ax = fig.add_subplot(gs[0, slot])
        im = ax.imshow(e["B"], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"B(r) for {e['label']}  (h,k) = ({e['h']}, {e['k']})\n"
                     f"azimuth {e['azimuth_deg']:.2f} deg   |q| = {e['q_nm_inv_abs']:.2f} nm^-1   "
                     f"rms = {e['rms_nm'] * 1e12:.1f} pm", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        axes.append(ax)
    cax = fig.add_axes([0.905, 0.15, 0.012, 0.7])
    fig.colorbar(im, cax=cax, label="B(r)  (same units as the input topography)")
    fig.suptitle(
        f"three r3 B(r) maps  |  local-q-map, lambda = {lam_nm:.0f} nm "
        f"(sigma_q = {sigma_q_px:.3f} px)  |  {STEM}_corrected  "
        f"({n} px, {fov_nm:.4f} nm)  |  shared absolute colour scale "
        f"+/-{vmax:.3e}", fontsize=12)
    out_three = HERE / f"B_r3_three_pairs_{FIGURE_TAG}.png"
    fig.savefig(out_three, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "input_h5": str(H5),
        "skill_report": str(REPORT),
        "lambda_nm": lam_nm,
        "sigma_q_px": sigma_q_px,
        "canvas_px": n,
        "field_of_view_nm": fov_nm,
        "colour_scale": {"vmin": vmin, "vmax": vmax, "rule": "p99.5 of |B| pooled over the three maps"},
        "colour_scale_note": "shared absolute scale across the three panels",
        "paired_by": "Friedel pair of each q (the -q partner is inside the same complex field)",
        "entries": [{k: v for k, v in e.items() if k != "B"} for e in entries],
        "three_panel_png": str(out_three),
    }
    (HERE / "B_r3_pairs.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    print(f"colour scale +/-{vmax:.6e}   sigma_q = {sigma_q_px:.4f} px")
    for e in entries:
        print(f"{e['label']} (h,k)=({e['h']},{e['k']}) az={e['azimuth_deg']:7.2f} "
              f"|q|={e['q_nm_inv_abs']:6.2f} nm^-1 |psi|_med={e['psi_median_abs']:.4e} "
              f"rms={e['rms_nm']:.4e}  corr(direct)={e['crosscheck_corr_field_vs_direct']:.6f} "
              f"ratio={e['crosscheck_rms_ratio_direct_over_field']:.6f} "
              f"maxdiff={e['crosscheck_max_abs_diff']:.3e}")
    print(f"wrote {out_three}")


if __name__ == "__main__":
    main()
