"""Regression checks for the Nanonis SXM binary-offset fix (M15).

The old loader started reading the big-endian float payload by skipping
two lines after ":SCANIT_END:" and seeking two more bytes. The new loader
locates the Nanonis "\x1a\x04" marker directly before the payload instead
(and falls back to the legacy heuristic with a warning when the marker is
missing).

The user reported no issues with the current behaviour, so the top
priority is non-regression: for every real file, the new loader output
must be element-wise identical (including NaN positions) to the legacy
reference implementation. The script also compares against nanonispy and
writes one Z(forward) topography PNG per file for manual inspection.

Every real file carries the b"\x1a\x04" marker, so the fallback to the
legacy offset (skip two lines, seek two bytes) is additionally exercised
on marker-stripped copies written to tmp_verify/: the copy must load the
identical payload and emit the fallback warning.

Run from the repository root:

    .venv/bin/python scripts/regression/check_nanonis_sxm.py

Exits with a non-zero status when any check fails. The real data files are
only ever opened for reading.
"""

import logging
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
import nanonispy as nap
import numpy as np

from stm_data_processing.io.nanonis_loader import NanonisFileLoader
from stm_data_processing.utils.plot_funcs import subtractMeanPlane

# Real Nanonis scan files (read-only). Optional CLI arguments may override.
SXM_FILES = [
    "/Users/hunfen/Documents/论文/Si111_Pb_islands/raw_data/2025-07-09/topo0002.sxm",
    "/Users/hunfen/Documents/论文/Si111_Pb_islands/raw_data/2025-07-09/topo0007.sxm",
    "/Users/hunfen/Documents/论文/Si111_Pb_islands/raw_data/2025-07-09/topo0011.sxm",
    "/Users/hunfen/Documents/论文/Si111_Pb_islands/raw_data/2025-07-09/topo0013.sxm",
    "/Users/hunfen/Documents/论文/Si111_Pb_islands/raw_data/2025-07-09/topo0019.sxm",
    "/Users/hunfen/Documents/论文/Si111_Pb_islands/raw_data/2025-07-09/topo0020.sxm",
]

PNG_DIR = Path(__file__).resolve().parents[2] / "tmp_verify" / "m15_sxm"

# Marker-stripped copies that exercise the legacy-offset fallback path.
FALLBACK_DIR = Path(__file__).resolve().parents[2] / "tmp_verify" / "m15_fallback"


def old_sxm_raw(f_path: str) -> np.ndarray:
    """Legacy reference: skip 2 lines + seek 2 bytes, then read >f4 floats."""
    with Path.open(f_path, "rb") as f:
        for line in f:
            decoded_line = line.decode(encoding="utf-8", errors="replace")
            if ":SCANIT_END:" in decoded_line:
                break
        # Skip two lines after the header and move the file pointer
        for _ in range(2):
            f.readline()
        f.seek(2, 1)
        return np.fromfile(f, dtype=">f")


def _check_png_content(png_path: Path, expected_finite: float) -> None:
    """Verify a rendered topography PNG is not blank and matches finite ratio.

    Crops the axes interior (skipping the title band and the colour bar)
    and checks colormap-independent statistics: the non-white fraction must
    be non-zero, the pixel-value spread (std) must rule out a blank or flat
    rendering, and the finite-pixel fraction must roughly match the finite
    ratio of the underlying data. The NaN pixels are rendered as the neutral
    gray #808080; they are classified only approximately, so the check does
    not depend on the colormap having no gray midtones.
    """
    import matplotlib.image as mpimg

    rgb = mpimg.imread(png_path)  # (H, W, 4), values in [0, 1]
    h, w = rgb.shape[:2]
    crop = rgb[int(h * 0.15) : int(h * 0.92), int(w * 0.05) : int(w * 0.78), :3]
    r, g, b = crop[..., 0], crop[..., 1], crop[..., 2]
    near_white = (r > 0.92) & (g > 0.92) & (b > 0.92)
    coloured = ~near_white
    coloured_frac = float(coloured.mean())

    # Colormap-independent non-blank checks: a blank image (all white) has a
    # tiny non-white fraction, and a flat image (single solid colour, e.g. an
    # all-NaN rendering) has a tiny pixel std. Measured on real files: std is
    # 0.26-0.40 for any finite data and ~0.04 for an all-NaN flat rendering,
    # so 0.1 separates the two with a wide margin.
    data_std = float(np.std(crop[coloured])) if coloured.any() else 0.0
    assert coloured_frac > 0.05, (
        f"rendered image is blank: non-white fraction {coloured_frac:.3f}"
    )
    assert data_std > 0.1, (
        f"rendered image is flat: pixel std {data_std:.4f}"
    )

    # NaN pixels are rendered as neutral gray #808080; classify them only
    # approximately (antialiasing may produce near-gray pixels elsewhere).
    neutral = (
        (np.abs(r - g) < 0.08) & (np.abs(g - b) < 0.08) & (np.abs(r - b) < 0.08)
    )
    near_bad = neutral & (r > 0.42) & (r < 0.58)  # NaN gray #808080
    finite_pixels = coloured & (~near_bad)
    finite_frac = float(finite_pixels.mean())
    assert abs(finite_frac - expected_finite) < 0.3, (
        f"finite pixel fraction {finite_frac:.3f} does not match "
        f"finite ratio {expected_finite:.3f}"
    )
    print(
        f"  pixel self-check: finite pixels {finite_frac:.3f} vs "
        f"finite {expected_finite:.3f} OK (std {data_std:.4f})"
    )


def check_file(f_path: str) -> None:
    """Verify old/new loader equivalence, nanonispy agreement and render PNG."""
    name = Path(f_path).name
    print(f"--- {name} ---")

    loader = NanonisFileLoader(f_path)
    header = loader.header
    channels = loader.channels
    # Regression (t22): .data access must never mutate the raw payload.
    # Snapshot the pristine 1-D array first, then verify it is unchanged
    # after .data is read, and that a second .data read is idempotent.
    raw_new = loader._raw_data.copy()
    data_reshaped = loader.data
    assert np.array_equal(raw_new, loader._raw_data, equal_nan=True), (
        "accessing .data mutated _raw_data in place"
    )
    data_again = loader.data
    assert np.array_equal(data_reshaped, data_again, equal_nan=True), (
        "repeated .data access is not idempotent"
    )

    scan_pixels = header.get("SCAN_PIXELS", "").strip()
    scan_range = header.get("SCAN_RANGE", "").strip()
    scan_dir = header.get("SCAN_DIR", "").strip()
    bias_v = float(header.get("BIAS", "0"))
    print(f"  SCAN_PIXELS: {scan_pixels!r}")
    print(f"  SCAN_RANGE: {scan_range!r}")
    print(f"  SCAN_DIR: {scan_dir!r}")
    print(f"  BIAS: {bias_v:.6g} V")
    print(f"  channels ({len(channels)}): {channels}")
    print(f"  data size: {raw_new.size} floats")

    px, py = (int(v) for v in scan_pixels.split())
    expected_min = len(channels) * 2 * px * py
    assert raw_new.size >= expected_min, (
        f"data size {raw_new.size} < expected {expected_min}"
    )
    print(f"  expected size (channels x 2 x pixels): {expected_min}")

    # (a) element-wise equivalence with the legacy reference implementation
    raw_old = old_sxm_raw(f_path)
    assert raw_old.size == raw_new.size, (
        f"legacy/new size mismatch: {raw_old.size} != {raw_new.size}"
    )
    assert np.array_equal(raw_old, raw_new, equal_nan=True), (
        "legacy and new loader outputs differ element-wise"
    )
    print(f"  old vs new: identical ({raw_old.size} floats, equal_nan=True)")

    # (b) agreement with nanonispy
    scan = nap.read.Scan(f_path)
    signals = scan.signals
    nano_1d = np.concatenate(
        [
            np.concatenate(
                [signals[ch]["forward"].ravel(), signals[ch]["backward"].ravel()],
            )
            for ch in signals
        ],
    )
    assert nano_1d.size == raw_new.size, (
        f"nanonispy/new size mismatch: {nano_1d.size} != {raw_new.size}"
    )
    assert np.array_equal(raw_new, nano_1d, equal_nan=True), (
        "new loader and nanonispy raw payloads differ element-wise"
    )
    flipud = scan_dir == "up"
    max_diff = 0.0
    for i, ch in enumerate(channels):
        ours = data_reshaped[2 * i]
        theirs = signals[ch]["forward"]
        if flipud:
            ours = np.flipud(ours)
        finite = np.isfinite(ours) & np.isfinite(theirs)
        assert finite.any(), f"channel {ch!r} has no finite values"
        assert np.allclose(ours[finite], theirs[finite], equal_nan=True), (
            f"channel {ch!r} forward differs from nanonispy"
        )
        max_diff = max(max_diff, float(np.max(np.abs(ours[finite] - theirs[finite]))))
    print(f"  nanonispy: {len(channels)} channels match, max abs diff {max_diff:.3e}")

    # (d) Z(forward) topography PNG for manual inspection.
    # NaN-safe plotting (t23): plane subtraction uses only finite pixels
    # (plot_funcs, fixed in t21), colour range uses NaN-safe quantiles,
    # uncollected scan rows are rendered as a distinguishable neutral
    # gray, and the title carries the data completeness.
    z_idx = channels.index("Z")
    z_fwd = data_reshaped[2 * z_idx]
    if flipud:
        z_fwd = np.flipud(z_fwd)
    z_plane = subtractMeanPlane(z_fwd.astype(float))

    finite_mask = np.isfinite(z_plane)
    assert np.array_equal(finite_mask, np.isfinite(z_fwd)), (
        "plane subtraction changed the NaN mask"
    )
    finite_ratio = float(finite_mask.mean())
    full_rows = int(np.count_nonzero(finite_mask.all(axis=1)))
    n_rows = z_plane.shape[0]
    finite_std = float(np.nanstd(z_plane))
    assert np.isfinite(finite_std) and finite_std > 0, (
        "finite region has zero variance after plane subtraction"
    )

    vmin, vmax = np.nanpercentile(z_plane, [2.0, 98.0])
    cmap = plt.get_cmap("afmhot").copy()
    cmap.set_bad("#808080")  # uncollected (NaN) pixels: neutral gray

    rng = [float(v) * 1e9 for v in scan_range.split()]  # m -> nm
    title = (
        f"{name}\n{px}x{py} px | {rng[0]:.2f}x{rng[1]:.2f} nm | "
        f"bias {bias_v * 1e3:.1f} mV | dir {scan_dir} | "
        f"finite {finite_ratio * 100:.1f}% ({full_rows}/{n_rows} rows)"
    )
    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(
        z_plane, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax
    )
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    fig.colorbar(im, ax=ax, label="Z (m), plane-subtracted")
    fig.tight_layout()
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    out_png = PNG_DIR / f"{name}.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(
        f"  PNG saved: {out_png} "
        f"(finite {finite_ratio * 100:.1f}%, std {finite_std:.3e})"
    )

    # Pixel-stat self-check on the rendered image: crop the axes interior
    # (skip the title band and the colour bar), classify pixels as
    # data-coloured / NaN-gray / background-white and require the
    # data-coloured fraction to roughly match the finite ratio and to be
    # non-zero (no more all-blank images).
    _check_png_content(out_png, finite_ratio)


class _RecordCapture(logging.Handler):
    """Collect log messages emitted while the handler is attached."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record.getMessage())


def check_fallback_path(f_path: str) -> None:
    """Exercise the legacy-offset fallback on a marker-stripped copy.

    All real files carry the b"\x1a\x04" binary start marker, so the
    marker-locating path is what runs in production. To cover the fallback
    (legacy skip-two-lines + seek-two-bytes offset), copy the file to
    tmp_verify with the two marker bytes replaced by b"\x00\x00" and verify
    the loader still returns the identical payload - it can only do so via
    the fallback - and that the fallback warning was emitted. The real data
    directory is only ever read.
    """
    src = Path(f_path)
    name = src.name
    FALLBACK_DIR.mkdir(parents=True, exist_ok=True)
    dst = FALLBACK_DIR / name

    raw = src.read_bytes()
    header_end = raw.index(b":SCANIT_END:") + len(b":SCANIT_END:")
    line_end = raw.index(b"\n", header_end)
    marker_pos = raw.find(b"\x1a\x04", line_end)
    assert marker_pos >= 0, f"{name}: b'\x1a\x04' marker not found"
    assert raw[line_end + 1 : marker_pos] == b"\n\n", (
        f"{name}: unexpected bytes before the marker: {raw[line_end + 1 : marker_pos]!r}"
    )
    # Replace the two marker bytes in place (same length) so the header and
    # the payload offset are untouched.
    stripped = raw[:marker_pos] + b"\x00\x00" + raw[marker_pos + 2 :]
    assert b"\x1a\x04" not in stripped[line_end : line_end + 64], (
        f"{name}: marker still present in the probed window"
    )
    dst.write_bytes(stripped)

    capture = _RecordCapture()
    loader_logger = logging.getLogger("stm_data_processing.io.nanonis_loader")
    loader_logger.addHandler(capture)
    try:
        fallback_loader = NanonisFileLoader(str(dst))
    finally:
        loader_logger.removeHandler(capture)

    original_loader = NanonisFileLoader(f_path)
    assert np.array_equal(
        fallback_loader._raw_data, original_loader._raw_data, equal_nan=True
    ), f"{name}: fallback payload differs from the marker path"
    # The fallback output must also equal the legacy reference on the copy.
    assert np.array_equal(
        fallback_loader._raw_data, old_sxm_raw(str(dst)), equal_nan=True
    ), f"{name}: fallback payload differs from the legacy reference"
    assert any("falling back to the legacy offset" in msg for msg in capture.records), (
        f"{name}: fallback warning was not emitted"
    )
    print(
        f"  fallback path: marker stripped -> identical payload "
        f"({fallback_loader._raw_data.size} floats), legacy offset + warning OK"
    )


def main() -> int:
    """Run every regression check; return a process exit code."""
    files = SXM_FILES if len(sys.argv) == 1 else sys.argv[1:]
    for f_path in files:
        if not Path(f_path).is_file():
            print(f"missing real data file: {f_path}")
            return 2

    failed = 0
    checks = 0
    for f_path in files:
        for check, label in (
            (check_file, Path(f_path).name),
            (check_fallback_path, f"{Path(f_path).name} fallback path"),
        ):
            checks += 1
            try:
                check(f_path)
                print(f"[PASS] {label}")
            except Exception as exc:  # report every failing check
                failed += 1
                print(f"[FAIL] {label}: {exc}")
                traceback.print_exc()

    print(f"{checks - failed}/{checks} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
