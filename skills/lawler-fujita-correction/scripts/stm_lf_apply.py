"""Apply a fitted Lawler-Fujita correction bundle to another topography dataset.

Stage 2 of the two-stage workflow: ``stm_lf_correct.py --save-transform`` writes the
bundle -- the JSON with the lock-in geometry and the npz with the dense displacement
field ``u`` in nm on the reference grid -- and this script re-runs **only the warp**
on an arbitrary target dataset.  It does **not** re-detect peaks and does **not**
re-run any self-check: the field is taken as given and its provenance is copied into
the log and the report, so the caller stays responsible for knowing that the target
is the same scan.

Transfer rules encoded here:

* the target must be the **same scan**, i.e. the same field of view; a mismatch is
  reported as a WARNING (and recorded in the report) but is not fatal;
* a **different pixel count** is supported: ``u`` is physical, so it is interpolated
  onto the target's pixel coordinates (nm/px of each image); an identical grid is
  used verbatim, which makes this stage reproduce the fit stage bit for bit;
* the target **need not show a lattice** -- carrying the field onto simultaneously
  measured data is the point of the transfer;
* the validity mask travels with the field and is resampled the same way.

Preprocessing and geometry mirror the fit stage: the target matrix is loaded with the
same delimiter auto-detection, preprocessed as ``flipud(subtractMeanPlane(...))``,
and warped as ``corrected(r) = T(r + u(r))`` with the bundle's ``pad`` and ``order``;
the corrected field of view is ``L * n_out / n``.
``method == "identity_fallback"`` means the fit stage corrected nothing, so the target
is copied verbatim (``n_out = n``, field of view unchanged, no NaN added).

Outputs (same naming and plot style as the fit stage):
    <outdir>/<stem>_corrected.csv       corrected topography (square, NaN padded)
    <outdir>/<stem>_corrected_fft2.npy  complex FFT2 (complex128, fftshifted)
    <outdir>/<stem>_corrected.png       topography plot (gwyddion colormap)
    <outdir>/<stem>_corrected_fft.png   FFT plot (inferno, log, percentile norm)
    <outdir>/correction.log             the console report (both contract lines)
    <outdir>/apply_report.json          the same numbers, machine readable

Usage:
    cd /path/to/STM_DataProcessing
    MPLCONFIGDIR=<writable> PYTHONDONTWRITEBYTECODE=1 \
        .venv/bin/python <this script> INPUT.csv --transform BUNDLE.json \
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import lf_lib as lf  # noqa: E402

BUNDLE_SCHEMA = "lawler-fujita-correction-transform"
BUNDLE_SCHEMA_VERSION = 1
DEFAULT_FOV_TOLERANCE = 1e-3


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "input", help="topography matrix (whitespace, tab or comma separated)"
    )
    parser.add_argument(
        "--transform",
        required=True,
        help="bundle JSON written by stm_lf_correct.py --save-transform",
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
        "--fov-tol",
        type=float,
        default=DEFAULT_FOV_TOLERANCE,
        help="relative field-of-view mismatch that triggers a WARNING "
        f"(default {DEFAULT_FOV_TOLERANCE:g})",
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


def resolve_delimiter(raw):
    """Mirror of stm_lf_correct.resolve_delimiter."""
    if raw is None:
        return None
    return raw.replace("\\t", "\t").replace("\\s", " ")


def load_matrix(path, delimiter, loader):
    """Mirror of stm_lf_correct.load_matrix: square plain-text matrix."""
    data = loader(path) if delimiter is None else np.loadtxt(path, delimiter=delimiter)
    if data.ndim != 2 or data.shape[0] != data.shape[1]:
        raise ValueError(f"{path}: expected a square 2D matrix, got {data.shape}")
    return data


def load_bundle(path):
    """Read and validate the bundle; ``ValueError`` carries the reason."""
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(
            f"{path}: the bundle JSON must be an object, got {type(payload).__name__}"
        )
    if payload.get("schema") != BUNDLE_SCHEMA:
        raise ValueError(
            f"{path}: schema must be '{BUNDLE_SCHEMA}', got {payload.get('schema')!r}"
        )
    if payload.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: schema_version must be {BUNDLE_SCHEMA_VERSION}, got "
            f"{payload.get('schema_version')!r}"
        )
    for key in (
        "n_px_reference",
        "field_of_view_nm_reference",
        "pad",
        "order",
        "method",
    ):
        if key not in payload:
            raise ValueError(f"{path}: missing key {key!r}")
    fallback = (
        bool(payload.get("fallback", False))
        or payload.get("method") == "identity_fallback"
    )
    u_path = payload.get("u_field_file")
    field = None
    if not fallback:
        if not u_path or not Path(u_path).is_file():
            raise ValueError(f"{path}: the u-field npz is missing ({u_path!r})")
        with np.load(u_path) as handle:
            field = {
                "u_x": handle["u_x"],
                "u_y": handle["u_y"],
                "valid": handle["valid"].astype(bool),
            }
        if (
            field["u_x"].shape != field["u_y"].shape
            or field["valid"].shape != field["u_x"].shape
        ):
            raise ValueError(
                f"{u_path}: u_x, u_y and valid must share one shape, got "
                f"{field['u_x'].shape}, {field['u_y'].shape}, "
                f"{field['valid'].shape}"
            )
    return {"payload": payload, "fallback": fallback, "field": field, "u_path": u_path}


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, args.stm_lib)
    from stm_data_processing.utils.bragg_peak import compute_fft2, load_image
    from stm_data_processing.utils.plot_funcs import subtractMeanPlane

    csv_path = Path(args.input)
    bundle_path = Path(args.transform)
    outdir = Path(args.outdir) if args.outdir else csv_path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem
    log_lines = []

    def emit(text=""):
        print(text)
        log_lines.append(text)

    try:
        bundle = load_bundle(bundle_path)
    except (OSError, json.JSONDecodeError, ValueError, KeyError) as exc:
        print(f"ERROR: unusable bundle: {exc}", file=sys.stderr)
        return 2
    payload = bundle["payload"]

    topo = load_matrix(csv_path, resolve_delimiter(args.delimiter), load_image)
    topo = np.flipud(subtractMeanPlane(topo))  # first row = top scan line
    n = int(topo.shape[0])
    reference_fov = float(payload["field_of_view_nm_reference"])
    fov_mismatch = abs(float(args.size_nm) - reference_fov) / max(reference_fov, 1e-12)

    emit("# apply a fitted Lawler-Fujita correction bundle (stage 2)")
    emit(f"# input: {csv_path}")
    emit(
        f"# canvas {n} x {n} px, field of view {args.size_nm:g} nm "
        f"({args.size_nm / n:.6f} nm/px)"
    )
    emit(f"# bundle: {bundle_path} (schema {BUNDLE_SCHEMA} v{BUNDLE_SCHEMA_VERSION})")
    emit(
        f"#   source input: {payload.get('source_input')} "
        f"({payload['n_px_reference']} px, field of view {reference_fov:g} nm, n_out "
        f"{payload.get('n_out_reference')})"
    )
    emit(f"#   source report: {payload.get('source_report')}")
    emit(
        f"#   method={payload.get('method')}, fallback={bundle['fallback']}, "
        f"lambda={payload.get('lambda_nm')} nm, Q_a_px={payload.get('q_a_px')}, "
        f"Q_b_px={payload.get('q_b_px')}"
    )
    emit(
        f"#   pad={payload['pad']}, order={payload['order']}, mask coverage "
        f"{100 * float(payload.get('mask_coverage_fraction', 0.0)):.2f} %"
    )
    emit(
        "# the bundle is applied as given: no peak detection and no self-check are "
        "re-run on this dataset"
    )
    if fov_mismatch > args.fov_tol:
        emit(
            f"# WARNING: the target field of view {args.size_nm:g} nm differs from the "
            f"reference {reference_fov:g} nm by {100 * fov_mismatch:.3f} % (tolerance "
            f"{100 * args.fov_tol:.3f} %): the bundle belongs to another scan, so the "
            "transferred field is only meaningful if the two images really cover the "
            "same area"
        )

    rescaled = False
    u_stats = {}
    if bundle["fallback"]:
        emit(
            "# WARNING: the bundle carries method = identity_fallback: the fit stage "
            "found no usable 1x1 ring and corrected nothing. The input is copied "
            "verbatim with its own canvas and field of view; do not treat this output "
            "as a corrected image."
        )
        corrected = np.array(topo, dtype=float, copy=True)
        n_out, half, magnitude_px, size_out = n, 0, 0.0, float(args.size_nm)
    else:
        field = bundle["field"]
        u_nm = np.stack([field["u_x"], field["u_y"]]).astype(float)
        valid = field["valid"]
        u_nm, valid = lf.resample_field(u_nm, valid, reference_fov, n, args.size_nm)
        rescaled = u_nm.shape[1] != int(payload["n_px_reference"]) or (
            float(args.size_nm) != reference_fov
        )
        values = u_nm[:, valid] if bool(np.any(valid)) else np.zeros((2, 0))
        u_stats = {
            "u_x": {
                "rms_nm": float(np.sqrt(np.mean(values[0] ** 2)))
                if values.size
                else 0.0
            },
            "u_y": {
                "rms_nm": float(np.sqrt(np.mean(values[1] ** 2)))
                if values.size
                else 0.0
            },
        }
        corrected, n_out, half, magnitude_px = lf.warp_by_field(
            topo,
            u_nm,
            args.size_nm,
            valid=valid,
            pad=int(payload["pad"]),
            order=int(payload["order"]),
        )
        size_out = float(args.size_nm) * n_out / n
        emit(
            f"# u field: {u_nm.shape[1]} x {u_nm.shape[1]} on the target grid"
            + (
                " (rescaled from the reference grid)"
                if rescaled
                else " (identical grid, used verbatim)"
            )
        )

    nan_fraction = (
        1.0 - float(np.count_nonzero(np.isfinite(corrected))) / corrected.size
    )
    emit(
        f"# corrected canvas: {n_out} x {n_out} px, field of view {size_out:.4f} nm "
        f"({size_out / n_out:.6f} nm/px), NaN {100 * nan_fraction:.2f} %"
    )

    out_csv = outdir / f"{stem}_corrected.csv"
    np.savetxt(out_csv, corrected, delimiter=",", fmt="%.10e")
    fft2_corrected = compute_fft2(corrected, size_out, subtract_plane=False)
    out_fft2 = outdir / f"{stem}_corrected_fft2.npy"
    np.save(out_fft2, fft2_corrected)

    lf.setup_style()
    cmap, cmap_source = lf.load_colormap(args.stm_lib)
    emit(f"# colormap: {cmap_source}")
    cmap_bad = cmap.copy()
    cmap_bad.set_bad(color=lf.BAD_COLOR)
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
        "skill": "lawler-fujita-correction",
        "stage": "apply",
        "input": str(csv_path),
        "transform_file": str(bundle_path),
        "source_input": payload.get("source_input"),
        "source_report": payload.get("source_report"),
        "method": payload.get("method"),
        "fallback": bool(bundle["fallback"]),
        "lambda_nm": payload.get("lambda_nm"),
        "a_nm": payload.get("a_nm"),
        "q_a_px": payload.get("q_a_px"),
        "q_b_px": payload.get("q_b_px"),
        "gauge": payload.get("gauge"),
        "field_of_view_nm": float(args.size_nm),
        "reference_field_of_view_nm": reference_fov,
        "fov_mismatch_fraction": float(fov_mismatch),
        "fov_warning": bool(fov_mismatch > args.fov_tol),
        "n_px": n,
        "u_field_rescaled": bool(rescaled),
        "u_field_grid_px": (None if bundle["fallback"] else int(n)),
        "u_field_file": bundle["u_path"],
        "u_stats_nm": u_stats,
        "canvas_growth_px": int(half),
        "max_displacement_px": float(magnitude_px),
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
