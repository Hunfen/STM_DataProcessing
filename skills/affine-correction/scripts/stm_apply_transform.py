"""Apply a fitted affine-correction transform to another topography dataset.

Stage 2 of the two-stage workflow: ``stm_topo_correct.py --save-transform`` writes
the fitted stretch ``M`` (pure stretch, no rotation) together with the reference
canvas geometry and the fit provenance; this script re-runs **only the resampling
step** of the correction on an arbitrary target dataset.  It does **not** re-detect
peaks and does **not** re-run the anchor self-check: the transform is taken as
given and its provenance is copied into the log and the report, so the caller stays
responsible for knowing that the transform was fitted on a comparable scan.

Preprocessing and geometry:

* the target matrix is loaded exactly like the fit stage (same delimiter
  auto-detection) and preprocessed the same way --
  ``flipud(subtractMeanPlane(...))``, first row = top scan line;
* the array-space map is the same ``A = P M^-1 P`` (``P`` the axis swap), but the
  canvas side ``n_out`` and the offset are recomputed for the **target's** pixel
  count ``n`` (see :func:`_image_transform`, a local mirror of
  ``stm_data_processing.utils.bragg_peak.correct._image_transform``);
* the corrected field of view is ``L * n_out / n`` (the nm-per-pixel scale is kept);
* ``method == "identity_fallback"`` means the fit stage found no positive-definite
  stretch and corrected nothing, so the target is copied verbatim (``n_out = n``,
  field of view unchanged, no NaN added).

Outputs (same naming and plot style as the fit stage):
    <outdir>/<stem>_corrected.csv       corrected topography (square, NaN padded)
    <outdir>/<stem>_corrected_fft2.npy  complex FFT2 (complex128, fftshifted)
    <outdir>/<stem>_corrected.png       topography plot (gwyddion colormap)
    <outdir>/<stem>_corrected_fft.png   FFT plot (inferno, log, percentile norm)
    <outdir>/correction.log             the console report (contract line included)
    <outdir>/apply_report.json          the same numbers, machine readable

Usage:
    cd /path/to/STM_DataProcessing
    MPLCONFIGDIR=<writable> PYTHONDONTWRITEBYTECODE=1 \
        .venv/bin/python <this script> INPUT.csv --transform TRANSFORM.json \
        -L 50 -o OUT_DIR [--delimiter ','] [--stm-lib DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import correction_lib as cl  # noqa: E402

# Maps physical (x, y) order to array (row, col) order and back.
AXIS_SWAP = np.array([[0.0, 1.0], [1.0, 0.0]])
TRANSFORM_SCHEMA = "topo-correction-transform"
TRANSFORM_SCHEMA_VERSION = 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "input", help="topography matrix (whitespace, tab or comma separated)"
    )
    parser.add_argument(
        "--transform",
        required=True,
        help="transform JSON written by stm_topo_correct.py --save-transform",
    )
    parser.add_argument(
        "-L",
        "--size-nm",
        type=float,
        required=True,
        help="scan size of THIS input in nm (square image)",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        default=None,
        help="output directory (default: next to the input file)",
    )
    parser.add_argument(
        "--delimiter",
        default=None,
        help="column delimiter (default: auto-detect tab/comma/whitespace; "
        "the escape '\\t' is accepted)",
    )
    parser.add_argument(
        "--stm-lib",
        default="/Users/hunfen/Documents/GitHub/STM_DataProcessing/src",
        help="STM_DataProcessing src directory",
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# mirrored helpers (the two scripts stay independent of each other)
# --------------------------------------------------------------------------- #
def resolve_delimiter(raw):
    """Mirror of stm_topo_correct.resolve_delimiter."""
    if raw is None:
        return None
    return raw.replace("\\t", "\t").replace("\\s", " ")


def load_matrix(path, delimiter, loader):
    """Mirror of stm_topo_correct.load_matrix: square plain-text matrix."""
    data = loader(path) if delimiter is None else np.loadtxt(path, delimiter=delimiter)
    if data.ndim != 2 or data.shape[0] != data.shape[1]:
        raise ValueError(f"{path}: expected a square 2D matrix, got {data.shape}")
    return data


def _image_transform(stretch, n, pad):
    """Array matrix, canvas side and offset of the resampling.

    Mirror of ``stm_data_processing.utils.bragg_peak.correct._image_transform``
    (kept local so the skill does not reach into a private package symbol): the map
    is ``A = P M^-1 P``, the canvas contains every transformed corner of the
    ``n``-pixel input plus ``pad`` pixels and the offset aligns the two centres.
    """
    matrix = AXIS_SWAP @ np.linalg.inv(stretch) @ AXIS_SWAP
    corners = (
        np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]) * (n - 1) / 2.0
    )
    extent = np.abs(np.linalg.inv(matrix) @ corners.T).max(axis=1)
    n_out = 2 * (int(np.ceil(float(extent.max()))) + int(pad)) + 1
    offset = (n - 1) / 2.0 - matrix @ np.array([(n_out - 1) / 2.0, (n_out - 1) / 2.0])
    return matrix, n_out, offset


def load_transform(path):
    """Read and validate the transform JSON; ``ValueError`` carries the reason.

    Required: the schema string, ``schema_version == 1`` and a finite 2x2
    ``affine_q`` with positive determinant (the map has to be invertible).  ``pad``
    and ``order`` default to the fit package's own values, ``method`` decides the
    ``identity_fallback`` special case.
    """
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(
            f"{path}: the transform JSON must be an object, got "
            f"{type(payload).__name__}"
        )
    if payload.get("schema") != TRANSFORM_SCHEMA:
        raise ValueError(
            f"{path}: schema must be '{TRANSFORM_SCHEMA}', got "
            f"{payload.get('schema')!r}"
        )
    if payload.get("schema_version") != TRANSFORM_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: schema_version must be {TRANSFORM_SCHEMA_VERSION}, "
            f"got {payload.get('schema_version')!r}"
        )
    try:
        affine = np.asarray(payload.get("affine_q"), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{path}: affine_q is not a numeric 2x2 matrix ({exc})"
        ) from exc
    if affine.shape != (2, 2) or not np.all(np.isfinite(affine)):
        raise ValueError(
            f"{path}: affine_q must be a finite 2x2 matrix, got "
            f"{payload.get('affine_q')!r}"
        )
    determinant = float(np.linalg.det(affine))
    if determinant <= 0.0:
        raise ValueError(
            f"{path}: affine_q must have a positive determinant, got "
            f"det = {determinant!r}"
        )
    order = int(payload.get("order", 3))
    pad = int(payload.get("pad", 10))
    if not 0 <= order <= 5:
        raise ValueError(f"{path}: order must be within 0..5, got {order}")
    if pad < 0:
        raise ValueError(f"{path}: pad must not be negative, got {pad}")
    return {
        "affine_q": affine,
        "order": order,
        "pad": pad,
        "method": str(payload.get("method", "")),
        "fallback": bool(payload.get("fallback", False)),
        "n_labelled": int(payload.get("n_labelled", 0)),
        "anchor_ring": str(payload.get("anchor_ring", "")),
        "anchor_verdict": str(payload.get("anchor_verdict", "")),
        "stretch_scale_sqrt_det": float(payload.get("stretch_scale_sqrt_det", np.nan)),
        "source_report": str(payload.get("source_report", "")),
        "source_input": str(payload.get("source_input", "")),
        "n_px_reference": int(payload.get("n_px_reference", 0)),
        "n_out_reference": int(payload.get("n_out_reference", 0)),
        "field_of_view_nm_reference": float(
            payload.get("field_of_view_nm_reference", np.nan)
        ),
    }


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, args.stm_lib)
    from stm_data_processing.utils.bragg_peak import compute_fft2, load_image
    from stm_data_processing.utils.plot_funcs import subtractMeanPlane

    csv_path = Path(args.input)
    transform_path = Path(args.transform)
    outdir = Path(args.outdir) if args.outdir else csv_path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem
    log_lines = []

    def emit(text=""):
        print(text)
        log_lines.append(text)

    try:
        transform = load_transform(transform_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR: unusable transform: {exc}", file=sys.stderr)
        return 2

    topo = load_matrix(csv_path, resolve_delimiter(args.delimiter), load_image)
    topo = np.flipud(subtractMeanPlane(topo))  # first row = top scan line
    n = int(topo.shape[0])
    affine_q = transform["affine_q"]
    pad = transform["pad"]
    order = transform["order"]

    emit("# apply a fitted correction transform (topo-correction skill, stage 2)")
    emit(f"# input: {csv_path}")
    emit(
        f"# canvas {n} x {n} px, field of view {args.size_nm:g} nm "
        f"({args.size_nm / n:.6f} nm/px)"
    )
    emit(
        f"# transform: {transform_path} (schema {TRANSFORM_SCHEMA} v"
        f"{TRANSFORM_SCHEMA_VERSION})"
    )
    emit(f"#   source report: {transform['source_report']}")
    emit(
        f"#   source input: {transform['source_input']} "
        f"({transform['n_px_reference']} px, field of view "
        f"{transform['field_of_view_nm_reference']:g} nm, n_out "
        f"{transform['n_out_reference']})"
    )
    emit(
        f"#   method={transform['method']}, fallback={transform['fallback']}, "
        f"n_labelled={transform['n_labelled']}, anchor_ring="
        f"{transform['anchor_ring']}, anchor_verdict="
        f"{transform['anchor_verdict']}"
    )
    emit(f"#   fitted stretch M =\n{affine_q}")
    emit(
        f"#   pad={pad}, order={order}, stretch scale |det M|^(1/2) = "
        f"{transform['stretch_scale_sqrt_det']:.5f}"
    )
    emit(
        "# the transform is applied as given: no peak detection and no anchor "
        "self-check are re-run on this dataset"
    )

    if transform["method"] == "identity_fallback":
        # Same semantics as the package: nothing was solved, so nothing is remapped.
        emit(
            "# WARNING: the transform carries method = identity_fallback: the fit "
            "stage found no positive-definite stretch and corrected nothing. The "
            "input is copied verbatim with its own canvas and field of view; do not "
            "treat this output as a corrected image."
        )
        corrected = np.array(topo, copy=True)
        matrix, offset = np.eye(2), np.zeros(2, dtype=float)
        n_out, size_out = n, float(args.size_nm)
    else:
        matrix, n_out, offset = _image_transform(affine_q, n, pad)
        corrected = affine_transform(
            topo,
            matrix,
            offset=offset,
            output_shape=(n_out, n_out),
            order=order,
            mode="constant",
            cval=np.nan,
            prefilter=order > 1,
        )
        size_out = float(args.size_nm) * n_out / n

    valid_fraction = float(np.count_nonzero(np.isfinite(corrected))) / float(
        corrected.size
    )
    nan_fraction = 1.0 - valid_fraction
    emit(
        f"# corrected canvas: {n_out} x {n_out} px, field of view "
        f"{size_out:.4f} nm ({size_out / n_out:.6f} nm/px), "
        f"NaN {100 * nan_fraction:.2f} %"
    )

    out_csv = outdir / f"{stem}_corrected.csv"
    np.savetxt(out_csv, corrected, delimiter=",", fmt="%.10e")
    fft2_corrected = compute_fft2(corrected, size_out, subtract_plane=False)
    out_fft2 = outdir / f"{stem}_corrected_fft2.npy"
    np.save(out_fft2, fft2_corrected)

    cl.setup_style()
    cmap, cmap_source = cl.load_colormap(args.stm_lib)
    emit(f"# colormap: {cmap_source}")
    cmap_bad = cmap.copy()
    cmap_bad.set_bad(color=cl.BAD_COLOR)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(corrected, cmap=cmap_bad, origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    out_png = outdir / f"{stem}_corrected.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    magnitude = np.abs(fft2_corrected)
    lo, hi = np.percentile(magnitude, [5, 99.5])
    logged = np.log(1.0 + magnitude)
    span = np.log(1.0 + hi) - np.log(1.0 + lo)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(
        np.clip((logged - np.log(1.0 + lo)) / span, 0.0, 1.0),
        cmap="inferno",
        origin="lower",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    out_fft_png = outdir / f"{stem}_corrected_fft.png"
    fig.savefig(out_fft_png, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    report = {
        "skill": "topo-correction",
        "stage": "apply",
        "input": str(csv_path),
        "transform_file": str(transform_path),
        "source_report": transform["source_report"],
        "source_input": transform["source_input"],
        "method": transform["method"],
        "fallback": transform["fallback"],
        "anchor_verdict": transform["anchor_verdict"],
        "field_of_view_nm": float(args.size_nm),
        "n_px": n,
        "affine_q": [[float(value) for value in row] for row in affine_q],
        "matrix_used": [[float(value) for value in row] for row in matrix],
        "offset": [float(value) for value in offset],
        "n_out": int(n_out),
        "corrected_field_of_view_nm": float(size_out),
        "corrected_nm_per_px": float(size_out / n_out),
        "nan_fraction": float(nan_fraction),
        "written": {
            "corrected_csv": str(out_csv),
            "corrected_fft2_npy": str(out_fft2),
            "corrected_png": str(out_png),
            "corrected_fft_png": str(out_fft_png),
            "correction_log": str(outdir / "correction.log"),
            "apply_report": str(outdir / "apply_report.json"),
        },
    }
    (outdir / "apply_report.json").write_text(json.dumps(report, indent=2) + "\n")
    emit("")
    emit(
        f"# written: {out_csv}, {out_fft2}, {out_png}, {out_fft_png}, "
        f"{outdir / 'apply_report.json'}, {outdir / 'correction.log'}"
    )
    (outdir / "correction.log").write_text("\n".join(log_lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
