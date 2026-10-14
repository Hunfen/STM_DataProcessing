# ruff: noqa
# fmt: off
# Byte-for-byte archive of the historical one-off lock-in script: kept as a record, never imported and not part of the runtime, so the body is frozen and both ruff gates are waived for this file.
"""Plot the demodulated phase maps theta(r) for q0, q1, q2.

Two versions, because the fields are almost uniform (R ~ 0.99, spread ~8 deg):
  absolute   theta in degrees, cyclic +/-180 (twilight)  -- shows it IS uniform
  relative   dtheta = wrap(theta - circular mean), one common window -- shows structure

Outputs (this folder):
  theta_q{0,1,2}_bare.png    1:1 pixel, twilight, common window, invalid pixels grey
  theta_maps_v1.png          2x3 annotated: top = absolute, bottom = relative
  theta_maps_v1.json         circular means, R, spreads, windows
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
from scipy.ndimage import distance_transform_edt

HERE = Path(__file__).resolve().parent
REPORT = HERE / "local_q_map_report.json"
TAG = "v1"


def grey_nan_cmap(name: str, colour: str = "0.55"):
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(colour)
    return cmap


def wrap_deg(x: np.ndarray) -> np.ndarray:
    return np.degrees(np.angle(np.exp(1j * np.radians(x))))


def main() -> None:
    rep = json.loads(REPORT.read_text(encoding="utf-8"))
    n = int(rep["canvas_px"])
    nm_per_px = float(rep["nm_per_px"])
    lam_nm = float(rep["lambda_nm"])
    border_px = 0.65 * lam_nm / nm_per_px

    entries = []
    for q in rep["q"]:
        with h5py.File(q["artifacts"]["h5"], "r") as f:
            theta = np.degrees(np.angle(np.exp(1j * f["theta"][:])))
            amp = f["amplitude"][:]
            mask = f["mask"][:]
        valid = mask > 0.5
        mean_deg = float(np.degrees(np.angle(np.mean(np.exp(1j * np.radians(theta[valid]))))))
        r_conc = float(np.abs(np.mean(np.exp(1j * np.radians(theta[valid])))))
        dtheta = wrap_deg(theta - mean_deg)
        entries.append({
            "label": q["label"], "h": q["h"], "k": q["k"],
            "azimuth_deg": float(np.degrees(np.arctan2(q["q_rad_px"][1], q["q_rad_px"][0])) % 360),
            "theta_circular_mean_deg": mean_deg,
            "resultant_R": r_conc,
            "circular_std_deg": float(np.degrees(np.sqrt(-2.0 * np.log(r_conc)))) if r_conc > 0 else float("nan"),
            "dtheta_rms_deg": float(np.sqrt(np.mean(dtheta[valid] ** 2))),
            "dtheta_p50_p99_5_deg": [float(np.percentile(np.abs(dtheta[valid]), 50)),
                                     float(np.percentile(np.abs(dtheta[valid]), 99.5))],
            "dtheta_min_max_deg": [float(dtheta[valid].min()), float(dtheta[valid].max())],
            "valid_fraction": float(valid.mean()),
            "theta": theta, "dtheta": dtheta, "valid": valid,
        })

    pooled = np.concatenate([np.abs(e["dtheta"][e["valid"]]) for e in entries])
    win = float(np.percentile(pooled, 99.5))
    win = float(np.ceil(win / 5.0) * 5.0)          # round up to a multiple of 5 deg
    summary = {
        "definition": "theta = arg psi_q (demodulated, no q.r ramp), low-passed by local-q-map "
                      "at lambda nm; plotted as degrees(angle(exp(i theta)))",
        "lambda_nm": lam_nm,
        "nm_per_px": nm_per_px,
        "border_erosion_px_for_mask_reference": border_px,
        "common_relative_window_deg": win,
        "window_rule": "p99.5 of |theta - circular mean| pooled over the three maps, "
                       "rounded up to 5 deg (so the three relative maps share one scale)",
        "entries": [{k: v for k, v in e.items() if k not in ("theta", "dtheta", "valid")}
                    for e in entries],
    }

    cmap = grey_nan_cmap("twilight")
    for e in entries:
        d = np.where(e["valid"], e["dtheta"], np.nan)
        fig = plt.figure(figsize=(n / 150.0, n / 150.0), dpi=150)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(d, origin="lower", cmap=cmap, vmin=-win, vmax=win)
        ax.set_axis_off()
        out = HERE / f"theta_{e['label']}_bare.png"
        fig.savefig(out, pad_inches=0)
        plt.close(fig)
        e["bare_png"] = str(out)

    fig = plt.figure(figsize=(19.5, 12.0), dpi=150)
    gs = fig.add_gridspec(2, 3, hspace=0.26, wspace=0.16)
    for slot, e in enumerate(entries):
        ax = fig.add_subplot(gs[0, slot])
        im = ax.imshow(np.where(e["valid"], e["theta"], np.nan), origin="lower", cmap=cmap,
                       vmin=-180, vmax=180)
        ax.set_title(f"{e['label']}  (h,k)=({e['h']:.3g},{e['k']:.3g})  az={e['azimuth_deg']:.2f}deg\n"
                     f"absolute theta, cyclic +/-180 deg   circular mean = "
                     f"{e['theta_circular_mean_deg']:.2f} deg", fontsize=10.5)
        ax.set_xticks([]); ax.set_yticks([])
        if slot == 2:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="theta (deg)")

        ax2 = fig.add_subplot(gs[1, slot])
        d = np.where(e["valid"], e["dtheta"], np.nan)
        im2 = ax2.imshow(d, origin="lower", cmap=cmap, vmin=-win, vmax=win)
        dist = distance_transform_edt(e["valid"])
        ax2.contour(np.where(dist > border_px, 1.0, 0.0), levels=[0.5], colors="w",
                    linewidths=0.5, alpha=0.7)
        ax2.set_title(f"{e['label']}: dtheta = theta - circular mean, common window +/-{win:.0f} deg\n"
                      f"R = {e['resultant_R']:.4f}  =>  circular spread "
                      f"{e['circular_std_deg']:.1f} deg;  dtheta rms {e['dtheta_rms_deg']:.2f} deg"
                      f"   (white line = 0.65 lambda border)", fontsize=10.5)
        ax2.set_xticks([]); ax2.set_yticks([])
        if slot == 2:
            fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02, label="dtheta (deg)")

    fig.suptitle(
        "demodulated phase of the three r3 components  |  local-q-map, lambda = "
        f"{lam_nm:.0f} nm  |  top: absolute (cyclic); bottom: relative to each circular mean, "
        f"shared window +/-{win:.0f} deg  |  grey = input NaN", fontsize=12)
    out = HERE / f"theta_maps_{TAG}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    summary["figure"] = str(out)
    (HERE / f"theta_maps_{TAG}.json").write_text(json.dumps(summary, indent=2, default=float),
                                                 encoding="utf-8")

    print(f"figure {out}   common relative window = +/-{win:.0f} deg")
    for e in entries:
        print(f"{e['label']} az={e['azimuth_deg']:7.2f}  circular mean = {e['theta_circular_mean_deg']:8.2f} deg  "
              f"R = {e['resultant_R']:.4f}  spread = {e['circular_std_deg']:5.1f} deg  "
              f"dtheta rms = {e['dtheta_rms_deg']:5.2f} deg  p50/p99.5 = "
              f"{e['dtheta_p50_p99_5_deg'][0]:.2f}/{e['dtheta_p50_p99_5_deg'][1]:.2f} deg  "
              f"range [{e['dtheta_min_max_deg'][0]:.1f},{e['dtheta_min_max_deg'][1]:.1f}]  "
              f"valid {e['valid_fraction'] * 100:.1f}%")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
