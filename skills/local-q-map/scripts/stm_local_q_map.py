"""Local-q-map of a corrected topography: Gaussian-windowed local Fourier maps.

For every requested q(h, k) of the ring_1x1 hexagonal reciprocal basis the script
demodulates the corrected canvas into one complex local field

    psi_q(r) = FFT^-1{ FFT[T(r) exp(-i q.r)] * exp(-Lambda^2 |k|^2 / 2) }

and writes, per q, the complex field (the primary product), its amplitude, its
demodulated phase ``theta_q = arg psi_q`` (radians, wrapped into ``(-pi, pi]``,
no ``q.r`` ramp) and the validity mask.  Everything is geometry and signal
processing: no physical statement is made anywhere.

Usage:
    cd /path/to/STM_DataProcessing
    MPLCONFIGDIR=<writable> PYTHONDONTWRITEBYTECODE=1 \
        .venv/bin/python <this script> CORRECTED.csv -o OUT_DIR \
        --basis-from correction_report.json --q 1,0 --q 0,1 --q 1/3,1/3 -L 50

    # explicit basis (signed fftshift offsets in px on the corrected canvas)
    .venv/bin/python <this script> CORRECTED.csv -o OUT_DIR \
        --basis-px '234.7,0;117.35,203.3' --q 1,0 --size-nm-from-log correction.log

A report fixes the RADIUS of the basis reliably, not its orientation: an affine
report's ``orientation_deg`` and a lawler-fujita report's lock-in directions both
live in the INPUT frame of the correction, while this script analyses the
corrected canvas.  The orientation is therefore measured on the corrected canvas
itself (``bragg_peak.detect_bragg_peaks`` over the ring_1x1 neighbourhood) and
replaces the report's value when the two disagree by more than 1 degree
(``--basis-orientation-from report`` forces the report frame back).  The resolved
basis is then validated against the canvas unconditionally: a mismatch is a
WARNING plus ``warnings.basis_ring_mismatch``, and the default ``--strict`` turns
it into exit 3 before any per-q artifact is written (``--no-strict`` keeps the
warning and the report flag only).

The rings are named

    ring_1x1   the reference ring: (h, k) = +/-(1, 0), (0, 1), (1, -1)
    ring_r3    the ring at 1/sqrt(3) of that radius: +/-(1/3, 1/3),
               (2/3, -1/3), (1/3, -2/3)

and are never named after a k-space high-symmetry point.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import localqmap as lq  # noqa: E402

SKILL_VERSION = lq.SKILL_VERSION
STM_LIB_DEFAULT = "/Users/hunfen/Documents/GitHub/STM_DataProcessing/src"

LOG: list[str] = []


def emit(message=""):
    """Print one log line and keep it for ``local_q_map.log``."""
    LOG.append(str(message))
    print(message)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "input", help="corrected topography CSV (square, may contain NaN)"
    )
    parser.add_argument(
        "-o",
        "--outdir",
        default=None,
        help="output directory (default: next to the input CSV)",
    )
    parser.add_argument(
        "--basis-from",
        default=None,
        help="correction report JSON; the ring_1x1 reciprocal basis is "
        "read from its keys (lawler-fujita style: q_a_nm_inv / "
        "q_b_nm_inv; affine style: orientation_deg plus "
        "b1_measured_nm_inv_after / b1_ideal_nm_inv)",
    )
    parser.add_argument(
        "--basis-px",
        default=None,
        help="escape hatch: explicit basis as signed fftshift offsets "
        "in px on the corrected canvas, 'b1x,b1y;b2x,b2y'",
    )
    parser.add_argument(
        "--q",
        dest="q_specs",
        action="append",
        default=None,
        help="q in basis coordinates, 'H,K' with arbitrary real numbers "
        "or fractions like '2/3,1/3'; repeatable, labelled q0, q1, "
        "... in the given order",
    )
    parser.add_argument(
        "-L",
        "--size-nm",
        type=float,
        default=None,
        help="field of view of the corrected canvas in nm",
    )
    parser.add_argument("--size-nm-from-log", default=None, help=lq.FOV_FROM_LOG_HELP)
    parser.add_argument(
        "--lambda-nm",
        type=float,
        default=lq.DEFAULT_LAMBDA_NM,
        help="Gaussian window width of the demodulation in nm "
        "(default 3.0 = the paper's 30 Angstrom; NOT the "
        "lawler-fujita-correction lambda, which defaults to 30 nm)",
    )
    parser.add_argument(
        "--amplitude-fraction",
        type=float,
        default=lq.DEFAULT_AMPLITUDE_FRACTION,
        help="mask threshold as a fraction of the median amplitude (default 0.10)",
    )
    parser.add_argument(
        "--basis-orientation-from",
        choices=("canvas", "report"),
        default="canvas",
        help="orientation of a report-based basis: 'canvas' (default) "
        "measures the ring_1x1 member azimuths of the corrected "
        "canvas (bragg_peak.detect_bragg_peaks) and applies them "
        "when they disagree with the report by more than "
        f"{lq.BASIS_RING_ORIENTATION_TOLERANCE_DEG:g} deg; "
        "'report' keeps the report's own orientation_deg / q_a "
        "direction, which for an affine report is the INPUT frame "
        "of the correction (use --basis-px if the corrected canvas "
        "is rotated against it)",
    )
    parser.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        default=True,
        help="DEFAULT ON: a basis-vs-canvas mismatch writes no per-q "
        "artifact and exits 3 (see --no-strict)",
    )
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="downgrade a basis-vs-canvas mismatch to the WARNING and the "
        "report flag only (exit 0, artifacts written)",
    )
    parser.add_argument(
        "--delimiter", default=",", help="delimiter of the corrected CSV (default ',')"
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="write the npy products only, no PNG preview",
    )
    parser.add_argument(
        "--stm-lib",
        default=STM_LIB_DEFAULT,
        help="STM_DataProcessing src directory (bragg_peak basis conventions)",
    )
    return parser.parse_args(argv)


def field_of_view(args, basis, n_px):
    """The corrected-canvas field of view in nm plus a one-line provenance string.

    Precedence: ``-L``, then ``--size-nm-from-log`` (the corrected-canvas line of
    the log wins, otherwise its last 'field of view <value> nm' line), then the
    correction report itself (its ``corrected_nm_per_px`` times the ``n_px``-wide
    canvas, else its ``corrected_field_of_view_nm``).  Neither present is an error.
    """
    if args.size_nm is not None:
        value = float(args.size_nm)
        return value, f"command line -L/--size-nm: {value:.4f} nm"
    if args.size_nm_from_log:
        value, source = lq.field_of_view_from_log(args.size_nm_from_log)
        return float(value), f"--size-nm-from-log: {value:.4f} nm ({source})"
    report_scale = basis.get("corrected_nm_per_px")
    if report_scale:
        per_px = float(report_scale)
        value = per_px * float(n_px)
        return value, (
            f"correction report corrected_nm_per_px {per_px:.8f} nm/px "
            f"times the {int(n_px)} px canvas"
        )
    report_fov = basis.get("corrected_field_of_view_nm")
    if report_fov:
        return float(report_fov), "correction report corrected_field_of_view_nm"
    raise SystemExit("a field of view is required: pass -L <nm> or --size-nm-from-log")


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, args.stm_lib)
    if (args.basis_from is None) == (args.basis_px is None):
        raise SystemExit(
            "exactly one basis source is required: --basis-from "
            "REPORT.json or --basis-px 'b1x,b1y;b2x,b2y'"
        )
    if not args.q_specs:
        raise SystemExit("at least one --q 'H,K' is required")

    csv_path = Path(args.input)
    outdir = Path(args.outdir) if args.outdir else csv_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    topo, nan_fraction = lq.load_topo(csv_path, args.delimiter)
    n = int(topo.shape[0])
    nan_region = ~np.isfinite(topo)

    basis = (
        lq.basis_from_report(args.basis_from)
        if args.basis_from
        else lq.basis_from_px(args.basis_px)
    )
    size_nm, fov_source = field_of_view(args, basis, n)
    nm_per_px = size_nm / n
    resolved = lq.resolve_basis(basis, n, nm_per_px)

    # the corrected canvas -- not the report -- decides the basis orientation: an
    # affine report's orientation_deg and an LF report's q_a both live in the INPUT
    # frame of the correction
    orientation = {
        "source": "command line --basis-px (explicit basis)",
        "applied": False,
        "delta_deg": 0.0,
        "canvas": None,
    }
    if resolved["kind"] in (
        "correction_report_affine",
        "correction_report_lawler_fujita",
    ):
        if args.basis_orientation_from == "report":
            orientation["source"] = (
                "report orientation (forced by --basis-orientation-from report)"
            )
        else:
            canvas_orientation = lq.canvas_ring_orientation(
                topo,
                size_nm,
                float(np.hypot(*resolved["b1_px"])),
                float(resolved["b1_angle_deg"]),
            )
            orientation["canvas"] = canvas_orientation
            if canvas_orientation["status"] != "measured":
                orientation["source"] = (
                    f"report orientation (the canvas could not "
                    f"be measured: {canvas_orientation['status']})"
                )
            elif (
                abs(float(canvas_orientation["delta_deg"]))
                > lq.BASIS_RING_ORIENTATION_TOLERANCE_DEG
            ):
                basis = lq.rebase_orientation(
                    basis,
                    canvas_orientation["delta_deg"],
                    "orientation from the corrected canvas: "
                    + canvas_orientation["message"]
                    + "; the report orientation is replaced by the canvas "
                    "measurement (rotation applied "
                    f"{float(canvas_orientation['delta_deg']):+.6f} deg)",
                )
                resolved = lq.resolve_basis(basis, n, nm_per_px)
                orientation.update(
                    source="corrected canvas (bragg_peak.detect_bragg_peaks)",
                    applied=True,
                    delta_deg=float(canvas_orientation["delta_deg"]),
                )
            else:
                orientation.update(
                    source="report orientation (the canvas agrees within "
                    f"{lq.BASIS_RING_ORIENTATION_TOLERANCE_DEG:g} deg)",
                    delta_deg=float(canvas_orientation["delta_deg"]),
                )

    b1_rad_px = np.asarray(resolved["b1_rad_px"], dtype=float)
    b2_rad_px = np.asarray(resolved["b2_rad_px"], dtype=float)
    basis_nm_per_px = float(resolved["basis_nm_per_px"])
    nm_per_px_ratio = basis_nm_per_px / nm_per_px
    nm_per_px_mismatch = bool(abs(nm_per_px_ratio - 1.0) > lq.NM_PER_PX_REL_TOLERANCE)

    specs = [lq.parse_q_spec(text) for text in args.q_specs]
    vectors = [lq.q_vector(h, k, b1_rad_px, b2_rad_px) for h, k in specs]
    width_px = lq.lambda_px(args.lambda_nm, nm_per_px)

    # unconditional basis-vs-canvas validation (both branches and --basis-px)
    canvas_check = lq.basis_canvas_check(
        topo,
        lq.ring_1x1_members_rad_px(b1_rad_px, b2_rad_px),
        float(np.hypot(*resolved["b1_px"])),
        n,
        width_px,
    )

    cross_talk = lq.pair_cross_talk(vectors, args.lambda_nm, nm_per_px)
    boundary_warning = bool(float(args.lambda_nm) > float(size_nm) / 4.0)
    border_band_px = lq.BOUNDARY_MARGIN_FACTOR * width_px

    # ------------------------------------------------------------------ header
    emit(f"# local-q-map v{SKILL_VERSION}")
    emit(f"# input: {csv_path}")
    emit(
        f"# canvas {n} x {n} px, field of view {size_nm:.4f} nm "
        f"({nm_per_px:.6f} nm/px), NaN {100.0 * nan_fraction:.2f} %"
    )
    emit(f"# field of view source: {fov_source}")
    emit(
        f"# basis source: {resolved['kind']}"
        + (f" ({resolved['path']})" if resolved.get("path") else "")
    )
    emit(f"# basis keys used: {', '.join(resolved['detected_keys'])}")
    for note in resolved["notes"]:
        emit(f"# basis note: {note}")
    emit(
        f"# basis b1 = ({resolved['b1_nm_inv'][0]:.6f}, {resolved['b1_nm_inv'][1]:.6f}) "
        f"nm^-1 = ({resolved['b1_rad_px'][0]:.8f}, {resolved['b1_rad_px'][1]:.8f}) "
        f"rad/px = ({resolved['b1_px'][0]:.4f}, {resolved['b1_px'][1]:.4f}) px, "
        f"|b1| = {resolved['abs_b1_nm_inv']:.6f} nm^-1 at "
        f"{resolved['b1_angle_deg']:.4f} deg"
    )
    emit(
        f"# basis b2 = ({resolved['b2_nm_inv'][0]:.6f}, {resolved['b2_nm_inv'][1]:.6f}) "
        f"nm^-1 = ({resolved['b2_rad_px'][0]:.8f}, {resolved['b2_rad_px'][1]:.8f}) "
        f"rad/px = ({resolved['b2_px'][0]:.4f}, {resolved['b2_px'][1]:.4f}) px at "
        f"{resolved['b2_angle_deg']:.4f} deg; angle between b1 and b2 = "
        f"{resolved['angle_between_deg']:.4f} deg (60 deg expected)"
    )
    emit(
        f"# basis nm/px used for the rad/nm -> rad/px conversion: "
        f"{resolved['basis_nm_per_px']:.8f} ({resolved['basis_conversion']})"
    )
    emit(f"# basis orientation source: {orientation['source']}")
    if orientation.get("canvas") is not None:
        emit(f"# basis orientation: {orientation['canvas']['message']}")
        if orientation["applied"]:
            emit(
                f"# basis orientation: the report orientation is replaced by the "
                f"corrected canvas (rotation {orientation['delta_deg']:+.6f} deg "
                "applied, radius unchanged)"
            )
    if canvas_check["mismatch"]:
        emit(f"# WARNING: basis-vs-canvas check: MISMATCH -- {canvas_check['reason']}")
    else:
        emit(f"# basis-vs-canvas check: {canvas_check['summary']}")
    if canvas_check["mismatch"] and args.strict:
        emit(
            "# ERROR: --strict (the default): the basis-vs-canvas check flagged a "
            "mismatch, no per-q artifact was written (exit 3; --no-strict downgrades "
            "this to a warning)"
        )
        (outdir / "local_q_map.log").write_text("\n".join(LOG) + "\n")
        return 3
    for text in resolved.get("warnings", []):
        emit(f"# WARNING: {text}")
    if nm_per_px_mismatch:
        emit(
            f"# WARNING: nm/px mismatch: the canvas gives L / N = {nm_per_px:.8f} "
            f"nm/px ({fov_source}) but the basis source converts with "
            f"{basis_nm_per_px:.8f} nm/px ({resolved['basis_conversion']}): ratio "
            f"{nm_per_px_ratio:.6f}. One of the two comes from another canvas: the "
            "reported q_nm_inv, lambda_px and the boundary band follow the canvas "
            "value, the basis follows the report"
        )
    if resolved.get("report_lockin_lambda_nm") is not None:
        emit(
            f"# basis-source report lockin.lambda_nm (reference only, NOT the window "
            f"of this skill): {resolved['report_lockin_lambda_nm']:.4f} nm"
        )
    emit(
        f"# window lambda = {args.lambda_nm:.4f} nm = {width_px:.4f} px, "
        f"amplitude fraction = {args.amplitude_fraction:.4f}"
    )
    emit(
        f"# boundary: the periodic FFT pollutes the demodulated maps within about "
        f"0.65 * lambda = {border_band_px:.4f} px of the border (reported, not cut)"
    )
    for record in cross_talk:
        if record["cross_talk"]:
            emit(
                f"# WARNING: q{record['q_i']} and q{record['q_j']} are "
                f"{record['separation_rad_per_nm']:.4f} nm^-1 apart, below the "
                f"2 / lambda = {record['threshold_rad_per_nm']:.4f} nm^-1 "
                "separation of the window: CROSS-TALK"
            )
    if boundary_warning:
        emit(
            f"# WARNING: lambda = {args.lambda_nm:.4f} nm exceeds L / 4 = "
            f"{size_nm / 4.0:.4f} nm: the window is not small against the field of "
            "view, the border pollution reaches most of the canvas"
        )

    # ------------------------------------------------------------ per q maps
    if not args.no_figures:
        lq.setup_style()
    entries = []
    written = []
    for index, ((h, k), vector) in enumerate(zip(specs, vectors, strict=True)):
        label = f"q{index}"
        q_px = lq.rad_px_to_px_offsets(vector, n)
        q_nm_inv = vector / nm_per_px
        field = lq.demodulate(topo, vector, args.lambda_nm, nm_per_px)
        amplitude = np.abs(field)
        valid, threshold = lq.valid_mask(amplitude, nan_region, args.amplitude_fraction)
        theta = np.angle(field)  # already in (-pi, pi]

        valid_fraction = float(np.mean(valid))
        if valid_fraction <= 0.0:
            emit(
                f"# WARNING: {label} has no valid pixel (amplitude threshold "
                f"{threshold:.6g}); its statistics are NaN"
            )
        amp_valid = amplitude[valid]
        amplitude_median, amplitude_fwhm = lq.linear_median_fwhm(amp_valid)
        mean_rad, resultant, _ = lq.circular_mean_rad(field, valid)
        median_rad, median_span_deg, n_minimisers = lq.circular_median_rad(
            theta[valid], weight=amp_valid
        )
        all_rad = float(np.angle(np.sum(field)))
        target = lq.demodulated_sum(topo, vector)
        identity_error_deg = float(
            np.degrees(lq.wrap_pm_pi(np.angle(np.sum(field)) - np.angle(target)))
        )

        stem = csv_path.stem
        prefix = f"{stem}_{label}"
        artifacts = {
            "field_npy": outdir / f"{prefix}_field.npy",
            "amplitude_npy": outdir / f"{prefix}_amplitude.npy",
            "theta_npy": outdir / f"{prefix}_theta.npy",
            "mask_npy": outdir / f"{prefix}_mask.npy",
        }
        if not args.no_figures:
            artifacts.update(
                amplitude_png=outdir / f"{prefix}_amplitude.png",
                theta_png=outdir / f"{prefix}_theta.png",
                mask_png=outdir / f"{prefix}_mask.png",
            )

        lq.save_npy(artifacts["field_npy"], np.asarray(field, dtype=np.complex128))
        lq.save_npy(artifacts["amplitude_npy"], np.asarray(amplitude, dtype=np.float64))
        lq.save_npy(artifacts["theta_npy"], np.asarray(theta, dtype=np.float64))
        lq.save_npy(artifacts["mask_npy"], np.asarray(valid, dtype=np.float64))
        if not args.no_figures:
            shown = np.where(valid, amplitude, np.nan)
            lq.save_map(artifacts["amplitude_png"], shown, cmap="inferno")
            wrapped_deg = np.degrees(np.angle(np.exp(1j * theta)))
            lq.save_map(
                artifacts["theta_png"],
                np.where(valid, wrapped_deg, np.nan),
                cmap="twilight",
                vmin=-180.0,
                vmax=180.0,
            )
            lq.save_map(
                artifacts["mask_png"],
                np.asarray(valid, dtype=np.float64),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
        written.extend(str(path) for path in artifacts.values())

        emit(
            f"# {label}: (h, k) = {lq.format_hk(h, k)} -> q_px "
            f"({q_px[0]:.4f}, {q_px[1]:.4f}), |q| = {float(np.hypot(*q_nm_inv)):.6f} "
            f"nm^-1, amplitude threshold {threshold:.6g}"
        )
        emit(
            f"# {label}: amplitude median {amplitude_median:.6g}, FWHM "
            f"{amplitude_fwhm:.6g}; theta circular mean "
            f"{np.degrees(mean_rad) % 360.0:.4f} deg, circular median "
            f"{np.degrees(median_rad) % 360.0:.4f} deg, R = {resultant:.6f}, "
            f"mean-identity error {identity_error_deg:.3e} deg"
        )
        emit(
            f"# {label}: mask coverage {100.0 * valid_fraction:.3f} % "
            f"(nan fraction {100.0 * nan_fraction:.3f} %)"
        )

        entries.append(
            {
                "label": label,
                "h": float(h),
                "k": float(k),
                "q_px": [float(value) for value in q_px],
                "q_rad_px": [float(value) for value in vector],
                "q_nm_inv": [float(value) for value in q_nm_inv],
                "q_nm_inv_abs": float(np.hypot(*q_nm_inv)),
                "lambda_nm": float(args.lambda_nm),
                "lambda_px": float(width_px),
                "amplitude_fraction": float(args.amplitude_fraction),
                "amplitude_threshold": float(threshold),
                "nan_fraction": float(nan_fraction),
                "mask_coverage_fraction": valid_fraction,
                "masked_fraction": float(1.0 - valid_fraction),
                "n_valid_pixels": int(np.count_nonzero(valid)),
                "amplitude": {
                    "median": float(amplitude_median),
                    "fwhm": float(amplitude_fwhm),
                    "max": float(np.max(amplitude)),
                    "min_valid": (
                        float(np.min(amp_valid)) if amp_valid.size else float("nan")
                    ),
                },
                "theta": {
                    "circular_mean_deg": float(np.degrees(mean_rad) % 360.0),
                    "circular_mean_all_pixels_deg": float(np.degrees(all_rad) % 360.0),
                    "mean_identity_error_deg": identity_error_deg,
                    "circular_median_deg": float(np.degrees(median_rad) % 360.0),
                    "circular_median_span_deg": float(median_span_deg),
                    "n_median_minimisers": int(n_minimisers),
                    "resultant_R": float(resultant),
                },
                "artifacts": {name: str(path) for name, path in artifacts.items()},
                "conventions": {
                    "field": "complex128 psi_q(r) = FFT^-1{FFT[T exp(-i q.r)] * "
                    "exp(-lambda^2 |k|^2 / 2)}; this is the primary product",
                    "amplitude": "|psi_q| ~ A_q / 2 for T containing A_q cos(q.r + phi_q)",
                    "theta": "arg psi_q in (-pi, pi] radians, demodulated convention "
                    "(carrier removed): theta ~ +phi_q, no q.r ramp; the "
                    "reported circular median is amplitude weighted (it "
                    "minimises sum_i |psi_i| |wrap(t - theta_i)|)",
                    "theta_png": "degrees(angle(exp(1j * npy))) in (-180, 180], twilight, "
                    "vmin=-180, vmax=180",
                    "mask": "float 1 = valid, 0 = invalid; invalid = input NaN region "
                    "UNION |psi_q| < amplitude_fraction * median(|psi_q|)",
                    "invalid_pixels": "drawn as NaN in the amplitude and theta PNGs "
                    "(bad colour), kept finite in every npy",
                },
            }
        )

    settings = {
        "skill": "local-q-map",
        "skill_version": SKILL_VERSION,
        "input": str(csv_path),
        "canvas_px": n,
        "field_of_view_nm": float(size_nm),
        "nm_per_px": float(nm_per_px),
        "field_of_view_source": fov_source,
        "nan_fraction": float(nan_fraction),
        "lambda_nm": float(args.lambda_nm),
        "lambda_px": float(width_px),
        "lambda_note": (
            "the window of this skill is a local Fourier window; it is NOT "
            "the lambda_nm of the lawler-fujita-correction skill"
        ),
        "amplitude_fraction": float(args.amplitude_fraction),
        "delimiter": args.delimiter,
        "figures": not args.no_figures,
        "basis_source": {
            "kind": resolved["kind"],
            "path": resolved.get("path"),
            "detected_keys": resolved["detected_keys"],
            "notes": resolved["notes"],
            "b1_nm_inv": resolved["b1_nm_inv"],
            "b2_nm_inv": resolved["b2_nm_inv"],
            "b1_rad_px": resolved["b1_rad_px"],
            "b2_rad_px": resolved["b2_rad_px"],
            "b1_px": resolved["b1_px"],
            "b2_px": resolved["b2_px"],
            "abs_b1_nm_inv": resolved["abs_b1_nm_inv"],
            "b1_angle_deg": resolved["b1_angle_deg"],
            "b2_angle_deg": resolved["b2_angle_deg"],
            "angle_between_deg": resolved["angle_between_deg"],
            "basis_nm_per_px": resolved["basis_nm_per_px"],
            "basis_conversion": resolved["basis_conversion"],
            "orientation_source": orientation["source"],
            "orientation_from_canvas": bool(orientation["applied"]),
            "orientation_delta_deg": float(orientation["delta_deg"]),
            "canvas_orientation": orientation["canvas"],
            "basis_canvas_check": canvas_check,
            "basis_warnings": resolved.get("warnings", []),
            "report_lockin_lambda_nm": resolved.get("report_lockin_lambda_nm"),
            "report_corrected_nm_per_px": resolved.get("corrected_nm_per_px"),
            "report_corrected_field_of_view_nm": resolved.get(
                "corrected_field_of_view_nm"
            ),
        },
        "warnings": {
            "cross_talk_pairs": cross_talk,
            "cross_talk": bool(any(item["cross_talk"] for item in cross_talk)),
            "boundary_warning": boundary_warning,
            "boundary_rule": "lambda > L / 4",
            "border_band_px": float(border_band_px),
            "border_band_rule": "0.65 * lambda",
            "basis_warnings": resolved.get("warnings", []),
            "basis_ring_mismatch": bool(canvas_check["mismatch"]),
            "basis_ring_mismatch_rule": (
                f"a ring_1x1 member below {lq.BASIS_RING_STRENGTH_FRACTION:g} of the "
                f"annulus peak, or a member offset above "
                f"{lq.BASIS_RING_OFFSET_FACTOR:g} window radii "
                f"({lq.BASIS_RING_OFFSET_FACTOR:g} * N / (2 pi lambda_px))"
            ),
            "basis_ring_strength_ratio": float(canvas_check["strength_ratio"]),
            "basis_ring_member_strength": float(canvas_check["member_strength"]),
            "basis_ring_annulus_strength": float(canvas_check["annulus_strength"]),
            "basis_ring_annulus_peak_px": canvas_check["annulus_peak_px"],
            "basis_ring_measured_angle_deg": canvas_check["measured_angle_deg"],
            "basis_ring_resolved_angle_deg": canvas_check["resolved_angle_deg"],
            "basis_ring_worst_offset_px": canvas_check["worst_offset_px"],
            "basis_ring_offset_tolerance_px": canvas_check["offset_tolerance_px"],
            "nm_per_px_mismatch": nm_per_px_mismatch,
            "nm_per_px_ratio": float(nm_per_px_ratio),
            "nm_per_px_mismatch_rule": (
                f"|basis_nm_per_px / (L / N) - 1| > {lq.NM_PER_PX_REL_TOLERANCE:g}"
            ),
            "canvas_nm_per_px": float(nm_per_px),
            "basis_nm_per_px": basis_nm_per_px,
        },
        "rings": {
            "ring_1x1_members_hk": [list(member) for member in lq.RING_1X1_MEMBERS],
            "ring_r3_members_hk": [list(member) for member in lq.RING_R3_MEMBERS],
            "naming": "rings are named ring_1x1 and ring_r3 only; no k-space "
            "high-symmetry name is used anywhere",
        },
        "conventions": {
            "grid": "r in [0, N)^2, array coordinates (row = y, col = x), "
            "nm/px = L / N",
            "basis": "{b1, b2} is the ring_1x1 hexagonal reciprocal basis, 60 deg "
            "apart (bragg_peak _HEXAGONAL_UNIT convention)",
            "q": "q(h, k) = h b1 + k b2 with arbitrary real h, k",
            "field": "psi_q(r) = FFT^-1{FFT[T(r) exp(-i q.r)] * "
            "exp(-lambda^2 |k|^2 / 2)}",
            "identities": [
                "sum_r psi_q = sum_r T exp(-i q.r) (exact, window independent)",
                "psi_{-q} = conj(psi_q) for real T (Friedel)",
                "psi_q(r - delta) = exp(-i q.delta) psi_q(r)",
            ],
        },
        "q": entries,
        "log": str(outdir / "local_q_map.log"),
        "report": str(outdir / "local_q_map_report.json"),
        "written": written,
    }
    report_path = outdir / "local_q_map_report.json"
    report_path.write_text(json.dumps(settings, indent=2) + "\n")
    emit("")
    emit(
        f"# written: {', '.join(written)}, {report_path}, {outdir / 'local_q_map.log'}"
    )
    (outdir / "local_q_map.log").write_text("\n".join(LOG) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
