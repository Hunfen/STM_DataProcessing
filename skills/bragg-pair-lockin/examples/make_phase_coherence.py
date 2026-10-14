# ruff: noqa
# fmt: off
# Byte-for-byte archive of the historical one-off lock-in script: kept as a record, never imported and not part of the runtime, so the body is frozen and both ruff gates are waived for this file.
"""Step 6 of notes/kekule_phase_extraction.md, run separately for q0, q1, q2.

C(dr) = < exp(i[theta(r) - theta(r+dr)]) > over valid pixel pairs

Computed as the circular autocorrelation of the unit field u = exp(i theta):
    A(dr) = IFFT( |FFT(u_masked)|^2 )        (numerator, zero outside the mask)
    N(dr) = IFFT( |FFT(mask)|^2   )          (number of valid pairs)
    C(dr) = A(dr) / N(dr)

Controls:
  positive  constant phase on the same mask          -> |C| = 1 exactly
  negative  phase randomised by block (block = lambda = 13 nm, offset 0/+-2pi/3)
            -> |C| must decay; the real data must not

The phase theta is the demodulated one (no q.r ramp) and is already low-passed by
local-q-map at lambda = 13 nm.  Only pixels farther than 0.65 lambda from any
invalid pixel are used, so the periodic-FFT border band is excluded.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.ndimage import distance_transform_edt

HERE = Path(__file__).resolve().parent
DATA = HERE.parent
REPORT = HERE / "local_q_map_report.json"
TAG = "v2"
BLOCK_NM = 13.0          # negative-control block size = the analysis resolution
MAX_LAG_NM = 60.0        # radial profile range
FIT_NM = 40.0            # exponential-fit range
RAMP_TEST_RAD_PER_NM = 0.01   # injected ramp for the estimator self-test


def radial_bins(cy: int, cx: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:n, 0:n]
    return np.hypot(yy - cy, xx - cx), np.hypot(yy - cy, 0)


def shell_profile(c: np.ndarray, w: np.ndarray, rad: np.ndarray, nm_per_px: float,
                  max_lag_nm: float) -> tuple[np.ndarray, np.ndarray]:
    """Pair-count-weighted shell mean of a COMPLEX correlation (real/imag binned apart)."""
    edges = np.arange(0.0, max_lag_nm / nm_per_px + 1.0, 1.0)
    idx = np.digitize(rad, edges) - 1
    keep = (idx >= 0) & (idx < len(edges) - 1) & (w > 0)
    k = idx[keep]
    ww = w[keep]
    nb = len(edges) - 1
    den_r = np.bincount(k, weights=ww, minlength=nb)
    re_r = np.bincount(k, weights=(np.real(c)[keep] * ww), minlength=nb)
    im_r = np.bincount(k, weights=(np.imag(c)[keep] * ww), minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        prof = (re_r + 1j * im_r) / den_r
    centres = 0.5 * (edges[:-1] + edges[1:]) * nm_per_px
    return centres, prof


def ramp_slopes(theta_field: np.ndarray, valid: np.ndarray, npair: np.ndarray,
                nm_per_px: float, xs_nm: np.ndarray, ys_nm: np.ndarray,
                max_nm: float = 50.0, n_lags: int = 400) -> dict:
    """Slope of arg C(dr) along +x and +y, i.e. the residual phase ramp of the field."""
    uu = np.where(valid, np.exp(1j * theta_field), 0j)
    spec = np.fft.fft2(uu)
    aa = np.fft.fftshift(np.fft.ifft2(spec * np.conj(spec)))
    with np.errstate(invalid="ignore", divide="ignore"):
        cc = np.where(npair > 0, aa / npair, np.nan)
    c = uu.shape[0] // 2
    lags = np.arange(n_lags) * nm_per_px
    out = {}
    for name, cut in (("x", cc[c, c:c + n_lags]), ("y", cc[c:c + n_lags, c])):
        ph = np.unwrap(np.angle(cut))
        good = np.isfinite(ph) & (lags <= max_nm)
        out[name] = float(np.polyfit(lags[good], ph[good], 1)[0]) if good.sum() > 5 else float("nan")
    return out


def main() -> None:
    rep = json.loads(REPORT.read_text(encoding="utf-8"))
    n = int(rep["canvas_px"])
    nm_per_px = float(rep["nm_per_px"])
    lam_nm = float(rep["lambda_nm"])
    border_px = 0.65 * lam_nm / nm_per_px
    block_px = BLOCK_NM / nm_per_px
    ys, xs = np.mgrid[0:n, 0:n]
    xs_nm, ys_nm = xs * nm_per_px, ys * nm_per_px

    fig = plt.figure(figsize=(19.5, 11.0), dpi=150)
    gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.22)
    out_entries = []

    for slot, q in enumerate(rep["q"]):
        with h5py.File(q["artifacts"]["h5"], "r") as f:
            theta = f["theta"][:]
            amp = f["amplitude"][:]
            mask = f["mask"][:]
        valid0 = mask > 0.5
        dist = distance_transform_edt(valid0)
        valid = valid0 & (dist > border_px)
        m = valid.astype(float)
        u = np.where(valid, np.exp(1j * theta), 0.0 + 0.0j)

        def corr(field: np.ndarray) -> np.ndarray:
            """Circular autocorrelation, COMPLEX (phase information must survive)."""
            spec = np.fft.fft2(field)
            return np.fft.fftshift(np.fft.ifft2(spec * np.conj(spec)))
        A = corr(u)
        N = np.real(corr(m))          # m is real, so this one is real
        with np.errstate(invalid="ignore", divide="ignore"):
            C = np.where(N > 0, A / N, np.nan)

        cy = cx = n // 2
        rad, rad1d = radial_bins(cy, cx, n)
        dr_nm, prof = shell_profile(C, N, rad, nm_per_px, MAX_LAG_NM)

        # --- controls -----------------------------------------------------
        rng = np.random.default_rng(20251112)
        nb = int(np.ceil(n / block_px))
        shifts = rng.choice([0.0, 2 * np.pi / 3, 4 * np.pi / 3], size=(nb, nb))
        shift_map = shifts[np.minimum((ys / block_px).astype(int), nb - 1),
                           np.minimum((xs / block_px).astype(int), nb - 1)]
        u_neg = np.where(valid, np.exp(1j * (theta + shift_map)), 0.0 + 0.0j)
        A_n = corr(u_neg)
        with np.errstate(invalid="ignore", divide="ignore"):
            C_n = np.where(N > 0, A_n / N, np.nan)
        dr_nm, prof_n = shell_profile(C_n, N, rad, nm_per_px, MAX_LAG_NM)

        u_pos = np.where(valid, 1.0 + 0.0j, 0.0 + 0.0j)
        A_p = corr(u_pos)
        with np.errstate(invalid="ignore", divide="ignore"):
            C_p = np.where(N > 0, A_p / N, np.nan)
        _, prof_p = shell_profile(C_p, N, rad, nm_per_px, MAX_LAG_NM)

        # --- exponential decay length -------------------------------------
        sel = (dr_nm > 0) & (dr_nm <= FIT_NM) & np.isfinite(np.abs(prof)) & (np.abs(prof) > 0)
        if sel.sum() >= 5:
            slope, intercept = np.polyfit(dr_nm[sel], np.log(np.abs(prof[sel])), 1)
            xi_nm = float(-1.0 / slope) if slope < 0 else float("inf")
        else:
            xi_nm = float("nan")
        sel_n = (dr_nm > 0) & (dr_nm <= FIT_NM) & np.isfinite(np.abs(prof_n)) & (np.abs(prof_n) > 0)
        xi_n = float(-1.0 / np.polyfit(dr_nm[sel_n], np.log(np.abs(prof_n[sel_n])), 1)[0]) \
            if sel_n.sum() >= 5 else float("nan")

        # --- directional 1D cuts: phase of C gives the (pseudo-gauge) ramp --
        slopes = ramp_slopes(theta, valid, N, nm_per_px, xs_nm, ys_nm)
        ramp = {name: {"slope_rad_per_nm": s, "omega_nm_inv": abs(s),
                       "strain_equivalent": abs(s) / 2.0 / 1.4082}  # 2A = 2*1.4082*eps
                for name, s in slopes.items()}
        # estimator truth test: inject a known ramp along x and along y; the
        # ESTIMATOR is linear, so the metric is the change relative to the
        # un-injected baseline (the real field has its own small arg-C slope).
        ramp_selftest = {}
        for tag, coords in (("inject_x", xs_nm), ("inject_y", ys_nm)):
            rec = ramp_slopes(theta + RAMP_TEST_RAD_PER_NM * coords, valid, N,
                              nm_per_px, xs_nm, ys_nm)
            dx = rec["x"] - slopes["x"]
            dy = rec["y"] - slopes["y"]
            ramp_selftest[tag] = {"injected_rad_per_nm": RAMP_TEST_RAD_PER_NM,
                                  "recovered_x": rec["x"], "recovered_y": rec["y"],
                                  "delta_x_vs_baseline": dx, "delta_y_vs_baseline": dy,
                                  "delta_ratio_x": dx / RAMP_TEST_RAD_PER_NM,
                                  "delta_ratio_y": dy / RAMP_TEST_RAD_PER_NM}

        prof_abs = np.abs(prof)
        out_entries.append({
            "label": q["label"], "h": q["h"], "k": q["k"],
            "azimuth_deg": float(np.degrees(np.arctan2(q["q_rad_px"][1], q["q_rad_px"][0])) % 360),
            "n_valid_px": int(valid.sum()), "valid_fraction": float(valid.mean()),
            "C_at_5nm": float(np.interp(5.0, dr_nm, prof_abs)),
            "C_at_13nm": float(np.interp(13.0, dr_nm, prof_abs)),
            "C_at_25nm": float(np.interp(25.0, dr_nm, prof_abs)),
            "C_at_50nm": float(np.interp(50.0, dr_nm, prof_abs)),
            "C_max_lag_nm": float(dr_nm[np.isfinite(prof_abs)][-1]),
            "C_at_max_lag": float(prof_abs[np.isfinite(prof_abs)][-1]),
            "xi_nm": xi_nm,
            "control_negative": {"block_nm": BLOCK_NM, "xi_nm": xi_n,
                                 "C_at_5nm": float(np.interp(5.0, dr_nm, np.abs(prof_n))),
                                 "C_at_13nm": float(np.interp(13.0, dr_nm, np.abs(prof_n))),
                                 "C_at_25nm": float(np.interp(25.0, dr_nm, np.abs(prof_n))),
                                 "C_at_50nm": float(np.interp(50.0, dr_nm, np.abs(prof_n)))},
            "control_positive": {"C_min": float(np.nanmin(np.abs(prof_p))),
                                 "C_max": float(np.nanmax(np.abs(prof_p)))},
            "ramp_from_argC": ramp,
            "ramp_estimator_selftest": ramp_selftest,
        })

        # --- top row: radial |C| ------------------------------------------
        ax = fig.add_subplot(gs[0, slot])
        ax.plot(dr_nm, prof_abs, "-", color="#1f77b4", lw=2.0, label="real q%s phase" % q["label"][-1])
        ax.plot(dr_nm, np.abs(prof_n), "--", color="#d62728", lw=1.6,
                label=f"control: registry shuffled per {BLOCK_NM:.0f} nm block")
        ax.plot(dr_nm, np.abs(prof_p), ":", color="0.35", lw=1.6, label="control: constant phase")
        ax.axhline(0.0, color="k", lw=0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)
        ax.set_xlabel("|dr| (nm)")
        ax.set_ylabel("|C(dr)|")
        ax.set_title(f"{q['label']}  (h,k)=({q['h']:.3g},{q['k']:.3g})  az={out_entries[-1]['azimuth_deg']:.2f}°\n"
                     f"xi = {xi_nm:.1f} nm   |C|(50 nm) = {out_entries[-1]['C_at_50nm']:.3f}   "
                     f"(control xi = {xi_n:.1f} nm)", fontsize=10.5)
        if slot == 0:
            ax.legend(fontsize=8.5, loc="upper right")

        # --- bottom row: 2D |C| -------------------------------------------
        ax2 = fig.add_subplot(gs[1, slot])
        s = int(60.0 / nm_per_px)
        sub = np.abs(C[cy - s:cy + s + 1, cx - s:cx + s + 1])
        im = ax2.imshow(sub, origin="lower", cmap="viridis", vmin=0, vmax=1,
                        extent=(-s * nm_per_px, s * nm_per_px, -s * nm_per_px, s * nm_per_px))
        ax2.plot([0], [0], "+", color="w", ms=6)
        ax2.set_xlabel("dx (nm)")
        ax2.set_ylabel("dy (nm)")
        ax2.set_title(f"{q['label']}: |C(dx,dy)|, centre = 0 lag", fontsize=10.5)
        if slot == 2:
            fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.02, label="|C|")

    fig.suptitle(
        "step 6: phase-difference coherence of the demodulated r3 fields "
        f"(local-q-map, lambda = {lam_nm:.0f} nm)  |  "
        f"mask eroded by 0.65 lambda = {border_px:.0f} px; radial profile to {MAX_LAG_NM:.0f} nm",
        fontsize=12)
    out_png = HERE / f"phase_coherence_C_{TAG}.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "definition": "C(dr) = <exp(i[theta(r)-theta(r+dr)])> over valid pairs; A/N normalisation",
        "lambda_nm": lam_nm, "nm_per_px": nm_per_px,
        "border_erosion_px": border_px, "max_lag_nm": MAX_LAG_NM, "fit_range_nm": FIT_NM,
        "criterion": "hidden order present <=> |C(dr)| stays close to 1 up to the field scale",
        "entries": out_entries,
        "figure": str(out_png),
    }
    (HERE / f"phase_coherence_C_{TAG}.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8")

    print(f"figure {out_png}")
    for e in out_entries:
        print(f"{e['label']} az={e['azimuth_deg']:7.2f} valid={e['valid_fraction'] * 100:5.1f}%  "
              f"|C|: 5nm {e['C_at_5nm']:.3f} 13nm {e['C_at_13nm']:.3f} 25nm {e['C_at_25nm']:.3f} "
              f"50nm {e['C_at_50nm']:.3f}  xi={e['xi_nm']:7.1f} nm | control xi={e['control_negative']['xi_nm']:5.1f} nm "
              f"|C|(50nm)={e['control_negative']['C_at_50nm']:.3f} | pos-control |C| in "
              f"[{e['control_positive']['C_min']:.6f},{e['control_positive']['C_max']:.6f}]")
        print(f"      argC ramp: x {e['ramp_from_argC']['x']['slope_rad_per_nm']:+.5f} rad/nm "
              f"y {e['ramp_from_argC']['y']['slope_rad_per_nm']:+.5f} rad/nm "
              f"(sigma-equivalent x/y {e['ramp_from_argC']['x']['strain_equivalent'] * 100:.3f}% / "
              f"{e['ramp_from_argC']['y']['strain_equivalent'] * 100:.3f}%)")
        st = e["ramp_estimator_selftest"]
        print(f"      ramp-estimator selftest (change vs un-injected baseline): "
              f"inject {st['inject_x']['injected_rad_per_nm']:.3f} rad/nm along x -> "
              f"dx {st['inject_x']['delta_x_vs_baseline']:+.5f} (ratio {st['inject_x']['delta_ratio_x']:+.4f}), "
              f"dy {st['inject_x']['delta_y_vs_baseline']:+.5f} (ratio {st['inject_x']['delta_ratio_y']:+.4f})")
        print(f"                                                        "
              f"inject    along y -> "
              f"dx {st['inject_y']['delta_x_vs_baseline']:+.5f} (ratio {st['inject_y']['delta_ratio_x']:+.4f}), "
              f"dy {st['inject_y']['delta_y_vs_baseline']:+.5f} (ratio {st['inject_y']['delta_ratio_y']:+.4f})")


if __name__ == "__main__":
    main()
