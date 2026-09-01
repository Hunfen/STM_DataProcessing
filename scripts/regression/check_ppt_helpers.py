"""Regression checks for the PPT helper bug fixes (M8, M9, M14, NaN handling).

Run from the repository root:
    .venv/bin/python scripts/regression/check_ppt_helpers.py

Covers:
  (a) subtractMeanPlane fitting correctness on a non-square 10x20 matrix;
  (b) bias label length vs data frame count for single- and multi-segment
      bias sweeps (including segment-boundary repeats), with the pre-fix
      behavior asserted for single segments;
  (c) static (AST) checks that the M8 fix reads raw_data_topo.header["bias"]
      and that M9/M14 reuse the shared plot_funcs helpers;
  (d) NaN-safe subtractMeanPlane: unmeasured (NaN) rows survive unchanged,
      the finite region keeps the exact plane fit, all-finite inputs match
      the pre-fix values bit-for-bit, degenerate inputs do not crash, and
      the NaN-aware colormap marks bad pixels.
"""

import ast
from pathlib import Path

import numpy as np

from stm_data_processing.utils.plot_funcs import (
    build_bias_labels,
    finite_range,
    subtractMeanPlane,
    topo_colormap,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUTOPPT_PATH = (
    PROJECT_ROOT / "src/stm_data_processing/utils/AutoPPt_winnew_modified.py"
)
PLOT_FUNCS_PATH = PROJECT_ROOT / "src/stm_data_processing/utils/plot_funcs.py"

SEG_BIAS_HEADER = (
    "Segment Start (V), Segment End (V), Settling (s), Integration (s), "
    "Steps (xn), Lockin, Init. Settling (s)"
)


def _buggy_subtract_mean_plane(matrix):
    """Exact copy of the pre-fix AutoPPt implementation (i*xdim+j stride)."""
    xdim, ydim = matrix.shape
    coord_matrix = np.zeros((xdim * ydim, 3))
    z_vector = np.zeros(xdim * ydim)
    for i in range(xdim):
        for j in range(ydim):
            coord_matrix[i * xdim + j] = [i, j, 1]
            z_vector[i * xdim + j] = matrix[i, j]
    plane_vector = np.linalg.inv(coord_matrix.T @ coord_matrix) @ coord_matrix.T @ z_vector
    plane_matrix = np.zeros((xdim, ydim))
    for i in range(xdim):
        for j in range(ydim):
            plane_matrix[i, j] = i * plane_vector[0] + j * plane_vector[1] + plane_vector[2]
    return matrix - plane_matrix


def test_subtract_mean_plane_non_square():
    """A known plane on a 10x20 (non-square) matrix must be removed exactly.

    The pre-fix stride (i*xdim+j) fits only a biased subset of the points and
    is exact for a pure plane, so the discriminating check uses a curved
    surface where the distorted fit must leave a measurably worse residual.
    """
    xdim, ydim = 10, 20
    y, x = np.meshgrid(np.arange(ydim), np.arange(xdim))
    coeffs = np.array([0.3, -0.2, 1.5])
    plane = coeffs[0] * x + coeffs[1] * y + coeffs[2]
    residual = subtractMeanPlane(plane)
    assert residual.shape == (xdim, ydim)
    max_res = np.max(np.abs(residual))
    assert max_res < 1e-8, f"plane not removed on non-square image, max residual {max_res}"
    curved = plane + 0.01 * (x - 4.5) * (y - 9.5)
    fixed_max = np.max(np.abs(subtractMeanPlane(curved)))
    try:
        buggy_max = np.max(np.abs(_buggy_subtract_mean_plane(curved)))
    except np.linalg.LinAlgError:
        buggy_max = float("inf")
    assert buggy_max > fixed_max * 1.5, (
        f"pre-fix stride must distort the fit: buggy {buggy_max} vs fixed {fixed_max}"
    )

def _pre_fix_subtract_mean_plane(matrix):
    """Reference copy of the pre-fix all-finite lstsq implementation."""
    xdim, ydim = matrix.shape
    y, x = np.meshgrid(np.arange(ydim), np.arange(xdim))
    A = np.column_stack([x.ravel(), y.ravel(), np.ones_like(x.ravel())])
    b = matrix.ravel()
    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    plane = coeffs[0] * x + coeffs[1] * y + coeffs[2]
    return matrix - plane


def test_subtract_mean_plane_nan_safe():
    """NaN rows must survive; the finite region keeps an exact plane fit.

    Mirrors an interrupted Nanonis scan: unmeasured rows are NaN, so only
    the finite pixels may define the fitted plane and the NaN pixels must
    stay NaN (never filled with the plane value).
    """
    xdim, ydim = 10, 20
    y, x = np.meshgrid(np.arange(ydim), np.arange(xdim))
    coeffs = np.array([0.3, -0.2, 1.5])
    plane = coeffs[0] * x + coeffs[1] * y + coeffs[2]
    mat = plane.copy()
    mat[:3, :] = np.nan  # 30% of the rows are unmeasured
    out = subtractMeanPlane(mat)
    # NaN positions and the finite mask are preserved pointwise.
    assert np.array_equal(np.isnan(out), np.isnan(mat))
    assert np.array_equal(np.isfinite(out), np.isfinite(mat))
    # The finite region lies on the known plane, so the residual is ~0.
    finite_mask = np.isfinite(mat)
    max_res = np.max(np.abs(out[finite_mask]))
    assert max_res <= 1e-10, f"finite residual too large: {max_res}"


def test_subtract_mean_plane_finite_identical_to_pre_fix():
    """All-finite inputs must be bit-identical to the pre-fix result."""
    rng = np.random.default_rng(11)
    mat = rng.normal(size=(12, 17))
    out = subtractMeanPlane(mat)
    ref = _pre_fix_subtract_mean_plane(mat)
    assert np.array_equal(out, ref), "finite path changed vs pre-fix values"


def test_subtract_mean_plane_degenerate_finite():
    """Fewer than 3 finite points returns a copy instead of crashing."""
    mat = np.full((4, 5), np.nan)
    mat[0, 0] = 1.0
    mat[1, 1] = 2.0  # only 2 finite points: a plane cannot be determined
    out = subtractMeanPlane(mat)
    assert np.array_equal(np.isnan(out), np.isnan(mat))
    assert out[0, 0] == 1.0 and out[1, 1] == 2.0
    all_nan = np.full((3, 3), np.nan)
    out_all = subtractMeanPlane(all_nan)
    assert np.all(np.isnan(out_all))


def test_topo_colormap_marks_bad_pixels():
    """The NaN-aware colormap must have an opaque, identifiable bad color."""
    cmap = topo_colormap("Blues_r")
    bad = cmap.get_bad()
    assert bad[3] == 1.0, "bad color must be opaque (identifiable)"
    # NaN-safe range helper must ignore NaN values.
    arr = np.array([[1.0, np.nan], [5.0, 3.0]])
    assert finite_range(arr) == (1.0, 5.0)
    assert finite_range(np.full((2, 2), np.nan)) == (None, None)



class FakeGrid:
    """Minimal stand-in for a nanonispy Grid (header + signals dicts)."""

    def __init__(self, header, signals):
        self.header = header
        self.signals = signals


def _seg_line(start, end, steps):
    """One Nanonis segment header entry (7 comma-separated fields)."""
    return f"{start},{end},0.1,0.01,{steps},1,0.1"


def test_single_segment_labels_match_pre_fix():
    """Single-segment labels must be identical to the pre-fix values."""
    steps, start, end = 10, -0.5, 0.5
    header = {SEG_BIAS_HEADER: [_seg_line(start, end, steps)]}
    sweep = np.linspace(start, end, steps)
    grid = FakeGrid(header, {"sweep_signal": sweep})
    for divider in (1, 10):
        labels = build_bias_labels(grid, divider)
        assert len(labels) == len(sweep)
        expected = np.linspace(start, end, steps) * 1000 / divider
        assert np.array_equal(labels, expected), "single-segment labels changed"


def test_multi_segment_keeps_boundary_repeats():
    """Axis with repeated segment-boundary points needs a full-length label list."""
    segs = [_seg_line(0.0, 1.0, 10), _seg_line(1.0, 2.0, 10)]
    sweep = np.concatenate([np.linspace(0.0, 1.0, 10), np.linspace(1.0, 2.0, 10)])
    grid = FakeGrid({SEG_BIAS_HEADER: segs}, {"sweep_signal": sweep})
    labels = build_bias_labels(grid, divider=1)
    assert len(labels) == len(sweep) == 20
    # Pre-fix this indexed into a 19-element list and raised IndexError at n=19.
    indexed = [labels[n] for n in range(len(sweep))]
    assert len(indexed) == len(sweep)
    assert labels[9] == labels[10] == 1000.0
    assert labels[19] == 2000.0


def test_multi_segment_deduped_axis_matches_pre_fix():
    """Axis without boundary repeats keeps the historical dedup behavior."""
    segs = [_seg_line(0.0, 1.0, 10), _seg_line(1.0, 2.0, 10)]
    sweep = np.concatenate([np.linspace(0.0, 1.0, 10), np.linspace(1.0, 2.0, 10)[1:]])
    grid = FakeGrid({SEG_BIAS_HEADER: segs}, {"sweep_signal": sweep})
    labels = build_bias_labels(grid, divider=1)
    assert len(labels) == len(sweep) == 19
    expected = (
        np.concatenate([np.linspace(0.0, 1.0, 10), np.linspace(1.0, 2.0, 10)[1:]])
        * 1000
    )
    assert np.array_equal(labels, expected), "deduped multi-segment labels changed"
    indexed = [labels[n] for n in range(len(sweep))]
    assert len(indexed) == len(sweep)


def test_missing_header_falls_back_to_sweep_signal():
    """Without the segment header the sweep signal itself provides the labels."""
    sweep = np.linspace(-0.2, 0.2, 7)
    grid = FakeGrid({}, {"sweep_signal": sweep})
    labels = build_bias_labels(grid, divider=1)
    assert len(labels) == len(sweep)
    assert np.array_equal(labels, sweep * 1000)


def test_malformed_header_raises_clear_error():
    """A length mismatch raises a descriptive ValueError, not an IndexError."""
    segs = [_seg_line(0.0, 1.0, 10), _seg_line(1.0, 2.0, 10)]
    sweep = np.linspace(0.0, 1.0, 17)  # matches neither 19 nor 20 frames
    grid = FakeGrid({SEG_BIAS_HEADER: segs}, {"sweep_signal": sweep})
    try:
        build_bias_labels(grid, divider=1)
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("expected ValueError for inconsistent label/axis length")


def _collect_raw_data_topo_bias_subscripts(node):
    """Return raw_data_topo.header["bias"] Subscript nodes under node."""
    found = []
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Subscript)
            and isinstance(sub.value, ast.Attribute)
            and sub.value.attr == "header"
            and isinstance(sub.value.value, ast.Name)
            and sub.value.value.id == "raw_data_topo"
            and isinstance(sub.slice, ast.Constant)
            and sub.slice.value == "bias"
        ):
            found.append(sub)
    return found


def test_m8_uses_raw_data_topo_bias():
    """Both setpointV_topo assignments must read raw_data_topo.header['bias']."""
    tree = ast.parse(AUTOPPT_PATH.read_text(encoding="utf-8"))
    assigns = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "setpointV_topo" for t in node.targets
        ):
            assigns.append(node)
    assert len(assigns) == 2, f"expected 2 setpointV_topo assignments, found {len(assigns)}"
    for node in assigns:
        expr = ast.unparse(node.value)
        names = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
        assert "raw_data_topo" in names, f"setpointV_topo must use raw_data_topo: {expr}"
        assert "raw_data" not in names, f"setpointV_topo must not use leftover raw_data: {expr}"
        assert _collect_raw_data_topo_bias_subscripts(node.value), (
            f"no raw_data_topo.header['bias'] subscript in: {expr}"
        )


def test_m9_and_m14_use_shared_helpers():
    """M9: AutoPPt no longer defines subtractMeanPlane; M14: map plotters use
    build_bias_labels."""
    tree = ast.parse(AUTOPPT_PATH.read_text(encoding="utf-8"))
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "subtractMeanPlane" not in funcs, "AutoPPt must reuse plot_funcs.subtractMeanPlane"
    for fname in ("ShowMap", "QPI", "ShowMapI"):
        body_names = {n.id for n in ast.walk(funcs[fname]) if isinstance(n, ast.Name)}
        assert "build_bias_labels" in body_names, f"{fname} must use build_bias_labels"
    plot_tree = ast.parse(PLOT_FUNCS_PATH.read_text(encoding="utf-8"))
    plot_funcs = {n.name: n for n in ast.walk(plot_tree) if isinstance(n, ast.FunctionDef)}
    for fname in ("plot_map_bias", "plot_qpi_bias", "plot_map_current_bias"):
        body_names = {n.id for n in ast.walk(plot_funcs[fname]) if isinstance(n, ast.Name)}
        assert "build_bias_labels" in body_names, f"{fname} must use build_bias_labels"


def main() -> None:
    tests = [
        test_subtract_mean_plane_non_square,
        test_subtract_mean_plane_nan_safe,
        test_subtract_mean_plane_finite_identical_to_pre_fix,
        test_subtract_mean_plane_degenerate_finite,
        test_topo_colormap_marks_bad_pixels,
        test_single_segment_labels_match_pre_fix,
        test_multi_segment_keeps_boundary_repeats,
        test_multi_segment_deduped_axis_matches_pre_fix,
        test_missing_header_falls_back_to_sweep_signal,
        test_malformed_header_raises_clear_error,
        test_m8_uses_raw_data_topo_bias,
        test_m9_and_m14_use_shared_helpers,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\nAll {len(tests)} PPT helper regression checks passed.")


if __name__ == "__main__":
    main()
