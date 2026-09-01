"""End-to-end verification of the Nanonis 3DS loader on real data (round 2).

Round 2 (t24) follows the user's explicit requirement: the dI/dV channel
is 'DSP 7280 Y (%)' (forward variant; the earlier round used LI Demod 1 X
and is superseded). Two real files are verified read-only:

- 1D line  'Grid Spectroscopy002.3ds' (2025-07-09, Grid dim 520 x 1,
  Points 401): dI/dV vs bias colormap (x = bias, y = position along the
  line, colour = dI/dV).
- 2D grid  'Grid Spectroscopy001.3ds' (2025-07-14, Grid dim 90 x 400,
  Points 7, 14 channels): one dI/dV map per bias slice (7 slices, each
  90 x 400) plus a multi-subplot overview; subplot titles carry the bias
  value rebuilt from the Sweep Start/End parameters.

The previous 2D verification used the incomplete 2025-10-24/008 file and
LI Demod 1 X; it is superseded by this round (its PNGs are removed).

Colour scales use NaN-safe quantiles; any uncollected pixels are rendered
as a distinguishable dark colour and the finite fraction is reported.

Run from the repository root:

    .venv/bin/python scripts/regression/check_3ds_real_data.py

Exits with a non-zero status when any check fails. Real data is read-only.
"""

import os
import sys
import tempfile
import traceback
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "dsh_mplconfig"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from stm_data_processing.io.nanonis_loader import NanonisFileLoader

# Real Nanonis files (read-only). Optional CLI arguments may override.
LINE_3DS = (
    "/Users/hunfen/Documents/论文/Si111_Pb_islands/raw_data/2025-07-09/"
    "Grid Spectroscopy002.3ds"
)
GRID_3DS = (
    "/Users/hunfen/Documents/论文/Si111_Pb_islands/raw_data/2025-07-14/"
    "Grid Spectroscopy001.3ds"
)

PNG_DIR = Path(__file__).resolve().parents[2] / "tmp_verify" / "m16_3ds"

HEADER_FIELDS = (
    "Grid dim",
    "Points",
    "Channels",
    "# Parameters (4 byte)",
    "Experiment size (bytes)",
)

# dI/dV channel per the user's explicit requirement (forward variant,
# i.e. the name without the "[bwd]" suffix).
DI_DV_CHANNEL = "DSP 7280 Y (%)"


def nan_quantile(data: np.ndarray, q: float) -> float:
    """NaN-safe percentile over the finite values of an array."""
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0
    return float(np.percentile(finite, q))


def load_and_report(path: str) -> tuple[dict, np.ndarray, object]:
    """Load a 3ds file, print key header fields and return its data."""
    loader = NanonisFileLoader(path)
    header = loader.header
    for field in HEADER_FIELDS:
        print(f"  {field}: {header.get(field)!r}")
    channels = loader.channels
    params = loader.parameters
    grid = loader.data
    print(f"  params shape: {params.shape}, columns: {list(params.columns)}")
    print(f"  grid shape: {grid.shape}, NaN ratio: {float(np.isnan(grid).mean()):.4f}")
    print(f"  channels ({len(channels)}): {channels}")
    ss = float(params["Sweep Start"].iloc[0])
    se = float(params["Sweep End"].iloc[0])
    print(f"  Sweep Start: {ss:.6g} V, Sweep End: {se:.6g} V")
    return header, grid, params


def rebuild_bias_axis(params: object, points: int) -> np.ndarray:
    """Rebuild the bias axis from Sweep Start/End; assert its length."""
    ss = float(params["Sweep Start"].iloc[0])
    se = float(params["Sweep End"].iloc[0])
    bias = np.linspace(ss, se, points)
    assert bias.size == points, f"bias axis length {bias.size} != Points {points}"
    print(
        f"  bias axis: {bias.size} points from {bias[0]:.6g} V to "
        f"{bias[-1]:.6g} V (Points = {points})"
    )
    return bias


def pick_didv(channels: list[str]) -> int:
    """Return the index of the forward 'DSP 7280 Y (%)' channel."""
    assert DI_DV_CHANNEL in channels, (
        f"{DI_DV_CHANNEL!r} missing from {channels}"
    )
    return channels.index(DI_DV_CHANNEL)


def check_1d_line(path: str) -> None:
    """Verify the 1D line file and render the dI/dV vs bias colormap."""
    print(f"--- 1D line: {Path(path).name} ---")
    header, grid, params = load_and_report(path)
    points = int(header["Points"])
    grid_dim = tuple(int(v) for v in header["Grid dim"].replace(" ", "").split("x"))
    channels = header["Channels"].split(";")
    assert min(grid_dim) == 1, f"not a line: Grid dim {grid_dim}"
    total_pixels = grid_dim[0] * grid_dim[1]
    expected_grid = (total_pixels, len(channels), points)
    assert grid.shape == expected_grid, (
        f"grid shape {grid.shape} != expected {expected_grid}"
    )
    assert params.shape[0] == total_pixels, (
        f"params rows {params.shape[0]} != pixels {total_pixels}"
    )
    assert params.shape[1] == int(header["# Parameters (4 byte)"]), (
        f"params columns {params.shape[1]} != param_length"
    )
    bias = rebuild_bias_axis(params, points)
    didv_idx = pick_didv(channels)

    line_data = grid[:, didv_idx, :]  # (n_pixels, points)
    finite = np.isfinite(line_data)
    print(f"  {DI_DV_CHANNEL} finite ratio: {finite.mean():.4f}")
    assert finite.any(), "line dI/dV data is entirely NaN"
    assert np.nanstd(line_data) > 0, "line dI/dV data is constant/zero"

    vmin = nan_quantile(line_data, 2.0)
    vmax = nan_quantile(line_data, 98.0)
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(
        line_data,
        aspect="auto",
        extent=[bias[0], bias[-1], 0, line_data.shape[0]],
        cmap="afmhot",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Bias (V)")
    ax.set_ylabel("Position (pixel)")
    ax.set_title(
        f"{Path(path).name}\ndI/dV ({DI_DV_CHANNEL}) vs bias | "
        f"Grid dim {grid_dim[0]}x{grid_dim[1]}",
        fontsize=10,
    )
    fig.colorbar(im, ax=ax, label="dI/dV (%)")
    fig.tight_layout()
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    out_png = PNG_DIR / "line002_didv_dsp7280y.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  dI/dV vs bias colormap saved: {out_png}")


def check_2d_grid(path: str) -> None:
    """Verify the 90x400 grid file and render per-bias-slice dI/dV maps."""
    print(f"--- 2D grid: {Path(path).name} ---")
    header, grid, params = load_and_report(path)
    points = int(header["Points"])
    grid_dim = tuple(int(v) for v in header["Grid dim"].replace(" ", "").split("x"))
    channels = header["Channels"].split(";")
    total_pixels = grid_dim[0] * grid_dim[1]

    expected_grid = (total_pixels, len(channels), points)
    assert grid.shape == expected_grid, (
        f"grid shape {grid.shape} != expected {expected_grid}"
    )
    assert params.shape[0] == total_pixels, (
        f"params rows {params.shape[0]} != pixels {total_pixels}"
    )
    assert params.shape[1] == int(header["# Parameters (4 byte)"]), (
        f"params columns {params.shape[1]} != param_length"
    )
    print(
        f"  shapes OK: grid={grid.shape}, params={params.shape}, "
        f"grid_dim={grid_dim}, points={points}"
    )
    bias = rebuild_bias_axis(params, points)
    didv_idx = pick_didv(channels)

    slices = grid[:, didv_idx, :]  # (total_pixels, points)
    finite_frac = np.isfinite(slices).mean()
    print(f"  {DI_DV_CHANNEL} finite ratio: {finite_frac:.4f}")
    assert finite_frac > 0.5, f"finite fraction {finite_frac:.3f} too low"

    # NaN-safe shared colour limits across all slices.
    vmin = nan_quantile(slices, 2.0)
    vmax = nan_quantile(slices, 98.0)
    cmap = plt.get_cmap("afmhot").copy()
    cmap.set_bad("#3a3a3a")  # distinguish uncollected (NaN) pixels

    maps = [slices[:, i].reshape(grid_dim) for i in range(points)]
    assert all(np.isfinite(m).any() for m in maps), "some slice is all-NaN"

    # Individual per-slice maps.
    for i, (bias_val, spec_map) in enumerate(zip(bias, maps, strict=True)):
        fig, ax = plt.subplots(figsize=(7, 5.5))
        im = ax.imshow(
            spec_map, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax
        )
        ax.set_title(
            f"{Path(path).name}\ndI/dV ({DI_DV_CHANNEL}) at bias "
            f"{bias_val * 1e3:.1f} mV (slice {i + 1}/{points})",
            fontsize=10,
        )
        ax.set_xlabel("x (pixel)")
        ax.set_ylabel("y (pixel)")
        fig.colorbar(im, ax=ax, label="dI/dV (%)")
        fig.tight_layout()
        out_png = PNG_DIR / f"grid001_didv_slice{i + 1}.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(
            f"  slice {i + 1} bias {bias_val * 1e3:.1f} mV: finite "
            f"{np.isfinite(spec_map).mean():.4f}, std "
            f"{np.nanstd(spec_map):.4g} -> {out_png.name}"
        )

    # Multi-subplot overview (2 rows x 4 columns, last cell unused).
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, (bias_val, spec_map) in enumerate(zip(bias, maps, strict=True)):
        ax = axes[i // 4, i % 4]
        im = ax.imshow(
            spec_map, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax
        )
        ax.set_title(f"bias {bias_val * 1e3:.1f} mV", fontsize=10)
        ax.set_xlabel("x (px)", fontsize=8)
        ax.set_ylabel("y (px)", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[1, 3].axis("off")
    fig.suptitle(
        f"{Path(path).name} dI/dV ({DI_DV_CHANNEL}) per bias slice",
        fontsize=12,
    )
    fig.colorbar(im, ax=axes, shrink=0.85, label="dI/dV (%)")
    fig.subplots_adjust(
        left=0.04, right=0.9, top=0.9, bottom=0.06, wspace=0.2, hspace=0.35
    )
    out_png = PNG_DIR / "grid001_didv_slices.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  overview saved: {out_png}")


def main() -> int:
    """Run every verification check; return a process exit code."""
    line_path = sys.argv[1] if len(sys.argv) > 1 else LINE_3DS
    grid_path = sys.argv[2] if len(sys.argv) > 2 else GRID_3DS
    for path in (line_path, grid_path):
        if not Path(path).is_file():
            print(f"missing real data file: {path}")
            return 2

    failed = 0
    checks = [
        ("1D line file (002.3ds)", lambda: check_1d_line(line_path)),
        ("2D grid file (001.3ds, 90x400)", lambda: check_2d_grid(grid_path)),
    ]
    for name, fn in checks:
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception as exc:  # report every failing check
            failed += 1
            print(f"[FAIL] {name}: {exc}")
            traceback.print_exc()

    total = len(checks)
    print(f"{total - failed}/{total} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
