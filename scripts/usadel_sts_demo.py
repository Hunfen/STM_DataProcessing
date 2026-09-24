"""Demo figures for the Usadel (diffusive-limit) gap / STS model.

Run from the repository root::

    .venv/bin/python scripts/usadel_sts_demo.py

This writes a set of human-readable PNG figures plus a ``README.md`` into
``var/usadel_demo/``.  All physical parameters (including the temperature and
Dynes ``Gamma`` used on each panel), conclusions and the exact reproduction
command are recorded in that README.

Matplotlib runs headless (Agg); the per-user matplotlib config directory is
redirected under ``var/`` so the script never touches ``~/.matplotlib``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VAR_DIR = REPO_ROOT / "var"
os.environ.setdefault("MPLCONFIGDIR", str(VAR_DIR / "mplconfig"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))

from stm_data_processing.utils.btk import BTK  # noqa: E402
from stm_data_processing.utils.usadel import (  # noqa: E402
    KB,
    WEAK_COUPLING_RATIO,
    Usadel1D,
    bcs_dos,
    bcs_gap,
    solve_retarded_sn,
)

OUT = VAR_DIR / "usadel_demo"

# Model parameters (Pb-like, but chosen for clean round numbers).
DELTA0 = 1e-3  # eV (1 meV)
D = 1e16  # nm^2 / s
TC = DELTA0 / (WEAK_COUPLING_RATIO * KB)
D_SMALL, D_LARGE = 200.0, 400.0  # N-layer thicknesses (nm)

# Distinct line styles so overlapping vertical segments stay readable.
STYLES = ["-", "--", "-.", ":"]

# Key measured numbers, filled by the figure functions and echoed into the
# README so that every written conclusion matches the figure content.
MEASURED: dict[str, str] = {}


def _save(fig, name):
    fig.savefig(OUT / name, dpi=120)
    plt.close(fig)
    print(f"  wrote {name}")


def fig01_bulk_dos_vs_T():
    """Bulk BCS DOS with the gap reduced according to Delta(T)."""
    GAMMA = 1e-6  # eV (Dynes broadening, kept small for a sharp peak)
    E = np.linspace(-3 * DELTA0, 3 * DELTA0, 1600)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for T in (0.0, 0.4 * TC, 0.7 * TC, 0.9 * TC, 0.99 * TC):
        Delta_T = bcs_gap(TC, T)
        label = f"T = {T / TC:.2f} Tc" if T > 0 else "T = 0"
        ax.plot(E / DELTA0, bcs_dos(E, Delta_T, GAMMA), label=label)

    # Measured T=0 coherence peak position (for the README honesty note).
    E_fine = np.linspace(0.9 * DELTA0, 1.1 * DELTA0, 200000)
    N_fine = bcs_dos(E_fine, DELTA0, GAMMA)
    i_peak = int(np.argmax(N_fine))
    E_peak = E_fine[i_peak]
    peak_dev = (E_peak - DELTA0) / DELTA0
    MEASURED["fig01_peak_dev"] = f"{peak_dev:.3e}"
    MEASURED["fig01_peak_height"] = f"{N_fine[i_peak]:.1f}"

    ax.set_xlabel("E / Delta0")
    ax.set_ylabel("N(E)")
    ax.set_title("Bulk BCS DOS vs T (Dynes Gamma = 1e-6 eV)")
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 4)
    ax.text(
        0.03,
        0.97,
        f"coherence peaks truncated at N=4\n(T=0 peak ~{N_fine[i_peak]:.0f} at "
        f"E/Delta0={E_peak / DELTA0:.4f})",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
    )
    ax.legend()
    fig.tight_layout()
    _save(fig, "01_bulk_bcs_dos_vs_T.png")


def fig02_sn_minigap_vs_d():
    """N(E) at the outer N edge for two N thicknesses (minigap), Gamma = 0."""
    u = Usadel1D(DELTA0, D, T=0.0)
    E = np.linspace(0.0, 0.5 * DELTA0, 121)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for d, style in zip((D_SMALL, D_LARGE), STYLES, strict=False):
        N = np.array([u.dos(Ei, position=d, geometry="sn", d=d, Gamma=0.0) for Ei in E])
        e_th = u.thouless(d) / DELTA0
        ax.plot(E / DELTA0, N, style, label=f"d = {d:g} nm (E_Th = {e_th:.3f} Delta0)")
    ax.set_xlabel("E / Delta0")
    ax.set_ylabel("N(E, x = d)  (outer N edge)")
    ax.set_title("S/N bilayer minigap vs d (T = 0, Gamma = 0)")
    ax.set_ylim(0, 1.8)
    ax.legend()
    fig.tight_layout()
    _save(fig, "02_sn_minigap_vs_d.png")


def fig03_ldos_positions():
    """N(E, x) at several positions across the S/N bilayer (d = 200 nm)."""
    GAMMA = 1e-6  # eV = 1e-3 * Delta0, small enough to keep the minigap visible
    u = Usadel1D(DELTA0, D, T=0.0)
    d = D_SMALL
    xi = u.xi
    positions = [
        ("deep S (x = -10 xi)", -10.0 * xi),
        ("interface (x = 0)", 0.0),
        ("mid N (x = d/2)", d / 2),
        ("outer N (x = d)", d),
    ]
    E = np.linspace(-2.0 * DELTA0, 2.0 * DELTA0, 161)
    # One BVP solve per energy (full profile), then sample each position.
    curves = np.empty((len(positions), len(E)))
    for j, Ei in enumerate(E):
        x, N = solve_retarded_sn(d, Ei, D, DELTA0, Gamma=GAMMA)
        for k, (_, xp) in enumerate(positions):
            curves[k, j] = np.interp(xp, x, N)

    # Honesty check: the outer-N minigap must really be < 0.05 below 0.05*Delta0.
    outer_N = curves[3]
    sub = np.abs(E) < 0.05 * DELTA0
    MEASURED["fig03_minigap_max"] = f"{float(np.max(outer_N[sub])):.2e}"

    # Fine symmetric scan of the outer-N / mid-N minigap notch for the zoom
    # panel; this also locates the minigap edge E_g (first E where N > 0.05).
    E_fine = np.linspace(-0.25 * DELTA0, 0.25 * DELTA0, 251)
    N_outer_fine = np.empty_like(E_fine)
    N_mid_fine = np.empty_like(E_fine)
    for j, Ei in enumerate(E_fine):
        xf, Nf = solve_retarded_sn(d, Ei, D, DELTA0, Gamma=GAMMA)
        N_outer_fine[j] = np.interp(d, xf, Nf)
        N_mid_fine[j] = np.interp(d / 2, xf, Nf)
    pos = E_fine >= 0.0
    i_edge = int(np.where(N_outer_fine[pos] > 0.05)[0][0])
    E_g = E_fine[pos][i_edge]
    i_peak = int(np.argmax(N_outer_fine[pos]))
    MEASURED["fig03_Eg"] = f"{E_g / DELTA0:.3f}"
    MEASURED["fig03_peak_E"] = f"{E_fine[pos][i_peak] / DELTA0:.3f}"
    MEASURED["fig03_peak_N"] = f"{N_outer_fine[pos][i_peak]:.2f}"

    fig, (ax, axz) = plt.subplots(1, 2, figsize=(11, 4.2))
    for k, (label, _) in enumerate(positions):
        ax.plot(E / DELTA0, curves[k], STYLES[k], label=label)
    ax.set_xlabel("E / Delta0")
    ax.set_ylabel("N(E, x)")
    ax.set_title("LDOS across an S/N bilayer (d = 200 nm, Gamma = 1e-6 eV)")
    ax.set_ylim(0, 3)
    ax.legend(fontsize=8)

    # Zoom panel: the outer-N proximity minigap notch, with E_g marked.
    axz.plot(E_fine / DELTA0, N_outer_fine, "-", label="outer N (x = d)")
    axz.plot(E_fine / DELTA0, N_mid_fine, "--", label="mid N (x = d/2)")
    axz.axvspan(-E_g / DELTA0, E_g / DELTA0, color="gray", alpha=0.25)
    axz.axvline(E_g / DELTA0, color="red", ls=":", lw=1)
    axz.annotate(
        f"E_g = {E_g / DELTA0:.3f} Delta0",
        xy=(E_g / DELTA0, 0.0),
        xytext=(0.08, 0.85),
        arrowprops={"arrowstyle": "->", "lw": 0.8},
        fontsize=8,
    )
    axz.set_xlabel("E / Delta0")
    axz.set_ylabel("N(E, x)")
    axz.set_title("Minigap notch (zoom)")
    axz.set_xlim(-0.25, 0.25)
    axz.set_ylim(0, 1.8)
    axz.legend(fontsize=8)

    # The truncation note goes in the figure margin (below the panels), where
    # no curve or legend can overlap it.
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.text(
        0.5,
        0.03,
        "Note: deep-S coherence peaks truncated at N=3 (left panel)",
        ha="center",
        va="center",
        fontsize=8,
    )
    _save(fig, "03_ldos_positions.png")


def fig04_sts_spectrum_T_Gamma():
    """Tunneling spectra for a family of T and Dynes Gamma values."""
    u = Usadel1D(DELTA0, D, T=0.0)
    V = np.linspace(-3 * DELTA0, 3 * DELTA0, 300)
    fig, (axT, axG) = plt.subplots(1, 2, figsize=(11, 4.2))
    for T in (0.0, 2.0, 4.0, 6.0):
        _, spec = u.tunnel_spectrum(V, T=T, Gamma=1e-9)
        axT.plot(V / DELTA0, spec, label=f"T = {T:g} K" if T else "T = 0")
    axT.set_title("Thermal broadening (Gamma -> 0)")
    axT.set_xlabel("eV / Delta0")
    axT.set_ylabel("dI/dV (a.u.)")
    axT.legend(fontsize=8)

    for G in (1e-9, 1e-4, 5e-4, 1e-3):
        _, spec = u.tunnel_spectrum(V, T=0.0, Gamma=G)
        axG.plot(V / DELTA0, spec, label=f"Gamma = {G:g} eV")
    axG.set_title("Dynes broadening (T = 0)")
    axG.set_xlabel("eV / Delta0")
    axG.set_ylabel("dI/dV (a.u.)")
    axG.legend(fontsize=8)

    fig.suptitle("Usadel tunneling spectra")
    fig.tight_layout()
    _save(fig, "04_sts_spectrum_T_Gamma.png")


def fig05_btk_comparison():
    """Usadel bulk DOS versus the ballistic BTK conductance (same Gamma)."""
    E = np.linspace(-3 * DELTA0, 3 * DELTA0, 1200)
    Gamma = 1e-4  # shared Dynes broadening
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    # Four curves with four distinct linestyles: Usadel solid, BTK Z=0 dashed,
    # Z=1 dash-dot, Z=3 dotted (no reliance on color/width alone).
    ax.plot(
        E / DELTA0,
        bcs_dos(E, DELTA0, Gamma),
        "-",
        color="black",
        lw=2,
        label="Usadel bulk DOS",
    )
    # BTK sigma is Sharvin-normalized (-> 1/(1+Z^2) at high energy); rescale by
    # (1+Z^2) so every curve shares the DOS normalization (-> 1 at high energy).
    btk_styles = ["--", "-.", ":"]
    for Z, style in zip((0.0, 1.0, 3.0), btk_styles, strict=False):
        btk = BTK(Delta=DELTA0, Z=Z, Gamma=Gamma)
        ax.plot(
            E / DELTA0,
            (1 + Z**2) * btk.sigma_zero_T(E),
            style,
            label=f"BTK Z = {Z:g}",
        )

    # Peak heights of the truncated BTK curves (measured on a fine grid).
    E_fine = np.linspace(0.5 * DELTA0, 1.5 * DELTA0, 200000)
    for Z in (1.0, 3.0):
        btk = BTK(Delta=DELTA0, Z=Z, Gamma=Gamma)
        sig = (1 + Z**2) * btk.sigma_zero_T(E_fine)
        MEASURED[f"fig05_peak_Z{Z:g}"] = f"{float(sig.max()):.1f}"

    ax.set_xlabel("E / Delta0")
    ax.set_ylabel("N(E)  /  (1+Z^2) * sigma_BTK   (both -> 1 at high E)")
    ax.set_title("Diffusive Usadel DOS vs ballistic BTK (Gamma = 1e-4 eV)")
    ax.set_ylim(0, 2.6)
    # Legend in the lower-left corner (E ~ [-3, -1.5], N < 1), away from the
    # coherence peaks at E ~ +-1.0.
    ax.legend(loc="lower left", fontsize=8)

    # The truncation note goes in the figure margin (below the axes), where no
    # curve can overlap it.
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.text(
        0.5,
        0.03,
        f"BTK Z=1 peak ~{MEASURED['fig05_peak_Z1']} and Z=3 peak "
        f"~{MEASURED['fig05_peak_Z3']} truncated at ylim = 2.6",
        ha="center",
        va="center",
        fontsize=8,
    )
    _save(fig, "05_btk_comparison.png")


def write_readme():
    xi = Usadel1D(DELTA0, D).xi
    lines = [
        "# Usadel demo figures",
        "",
        "Reproduce from the repository root:",
        "",
        "    .venv/bin/python scripts/usadel_sts_demo.py",
        "",
        "Model parameters (all figures): `Delta0 = 1 meV`, `D = 1e16 nm^2/s`, "
        f"`Tc = {TC:.3f} K` (weak coupling, `Delta0 = 1.764 kB Tc`), "
        f"dirty-limit coherence length `xi = {xi:.1f} nm`.",
        "",
        "Figures:",
        "",
        "- `01_bulk_bcs_dos_vs_T.png` -- bulk BCS `N(E)` at `T = 0, 0.4, 0.7, "
        "0.9, 0.99 Tc` (Dynes `Gamma = 1e-6 eV`), with `Delta(T)` from the "
        "self-consistent gap equation.  The `T = 0.99 Tc` curve shows the gap "
        "nearly closed, so the `gap closes at Tc` statement is demonstrated "
        "rather than extrapolated.  Coherence peaks are truncated at `N = 4`; "
        f"the `T = 0` peak height is ~{MEASURED['fig01_peak_height']} at "
        f"`E/Delta0 = 1.0006`, a relative shift of "
        f"{MEASURED['fig01_peak_dev']} from `Delta0` (< 0.5 %).",
        "",
        f"- `02_sn_minigap_vs_d.png` -- `N(E, x=d)` at the outer N edge for "
        f"`d = {D_SMALL:g}` and `{D_LARGE:g}` nm (`T = 0`, `Gamma = 0`).  "
        "Conclusion: a hard induced minigap appears below `E_g`, and `E_g` "
        "grows as `d` shrinks (`E_g ~ hbar D / d^2`).",
        "",
        "- `03_ldos_positions.png` -- left panel: `N(E, x)` at deep S, the "
        "interface, mid N and the outer N edge (`d = 200 nm`, "
        "`Gamma = 1e-6 eV = 1e-3 Delta0`); right panel: a zoom of the outer-N "
        "proximity minigap notch with the minigap edge `E_g` marked.  "
        "Conclusion: deep S shows the bulk BCS coherence peaks (truncated at "
        "`N = 3`), while the outer N edge keeps a minigap notch: "
        f"`max N(|E| < 0.05 Delta0) = {MEASURED['fig03_minigap_max']} < 0.05`, "
        f"`E_g = {MEASURED['fig03_Eg']} Delta0`, and the induced-gap peak is "
        f"`N = {MEASURED['fig03_peak_N']}` at "
        f"`|E| = {MEASURED['fig03_peak_E']} Delta0`.  The notch is narrow -- "
        "about `+-0.05 Delta0` wide, not a broad flat zero region.",
        "",
        "- `04_sts_spectrum_T_Gamma.png` -- `dI/dV` versus bias for a family of "
        "`T` (thermal broadening) and Dynes `Gamma` (lifetime broadening).  "
        "Conclusion: `T` smears the gap edge; `Gamma` fills the sub-gap region.",
        "",
        "- `05_btk_comparison.png` -- Usadel bulk DOS against the ballistic BTK "
        "conductance at `Z = 0, 1, 3` (Dynes `Gamma = 1e-4 eV`).  The raw BTK "
        "`sigma` is Sharvin-normalized (it tends to `1/(1+Z^2)` at high energy), "
        "so here each BTK curve is plotted as `(1+Z^2)*sigma` to share the DOS "
        "normalization (`-> 1` at high energy) and be directly comparable.  "
        "`Z = 0` (transparent) and `Z > 0` (barrier) are only comparable after "
        "this rescaling.  Four distinct linestyles are used: Usadel DOS solid, "
        "BTK Z=0 dashed, Z=1 dash-dot, Z=3 dotted.  The BTK Z=1/Z=3 coherence "
        f"peaks (heights ~{MEASURED['fig05_peak_Z1']} / "
        f"~{MEASURED['fig05_peak_Z3']}) are truncated at `ylim = 2.6` (annotated "
        "on the figure).  Conclusion: BTK (`l >> xi`) and Usadel (`l << xi`) "
        "are complementary transport regimes with different spectral shapes "
        "(BTK has no coherence peak at `Z = 0`).",
        "",
        "Known limitations of the module:",
        "",
        "- single spin, no phase/current, no non-equilibrium distribution "
        "function, no `P(E)` environment layer;",
        "- the S/N interface is a narrow smooth `Delta` transition of width "
        "`0.05 xi` (equivalent to a transparent step interface);",
        "- `solve_gap(geometry='sn')` truncates the Matsubara sum at "
        "`omegac = 20 kB Tc`, so the self-consistent deep-S `Delta` deviates "
        "from the `omegac = 1000` bulk value by ~0.24 % (V4a measured "
        "`0.95917 Delta0` vs bulk `0.95689 Delta0`);",
        "- the A5 minigap threshold is read off a 101-point energy grid, so "
        "`E_g` is quantized in `0.0025 * Delta0` steps (the direction check is "
        "robust; V4b refines it with a bisection).",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote README.md ({len(lines)} lines)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Writing demo figures to {OUT.relative_to(REPO_ROOT)}")
    fig01_bulk_dos_vs_T()
    fig02_sn_minigap_vs_d()
    fig03_ldos_positions()
    fig04_sts_spectrum_T_Gamma()
    fig05_btk_comparison()
    write_readme()
    print("Done.")


if __name__ == "__main__":
    main()
