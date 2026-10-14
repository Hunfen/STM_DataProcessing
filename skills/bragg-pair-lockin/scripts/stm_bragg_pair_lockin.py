"""Bragg-pair lock-in: arbitrary Bragg points -> local fields -> real-space maps.

For every requested point q(h, k) the script

1. computes the local (complex) field of q with the engine of the sibling skill
   ``local-q-map`` (imported in-process, never copied and never run as a
   subprocess) together with its amplitude, its demodulated phase and its
   validity mask,
2. checks the three applicability criteria of the analytic reconstruction
   ``B(r) = 2 Re[psi_q(r) exp(i q.r)]`` -- Friedel partner suppression
   ``2 (lambda |q|)^2``, cross-talk ``|dq|`` against ``2 / lambda`` and the
   envelope change per carrier cycle -- and switches that point to the strict
   two-sided reconstruction on the raw spectrum when one of them fails,
3. estimates the phase-difference coherence of that point's own field
   (``C(dr) = <exp(i[theta(r) - theta(r + dr)])>``, its decay length ``xi`` and
   the residual phase ramp of ``arg C`` along +x and +y),
4. writes the per-point HDF5 product and previews, the per-pair ``B(r)`` product
   and preview, the generalized five-step figures, a machine readable report and
   the log.

``--replot`` redraws every figure of an existing run from its products (report,
h5, and the CSV only for the canvas spectrum): it does not demodulate, does not
write an h5 and does not touch the report.

Nothing of this layer is statistical: cross-point integration, pairwise or
triple-product statistics and figure atlases belong to the sibling skill
``phase-analysis``.

Both spellings of a point are accepted and printed everywhere:

    --q 1/3,1/3     q = h b1 + k b2            (h, k) is the internal truth
    --g 3:1,1       q = (1 b1 + 1 b2) / 3

The point set must be given explicitly.  Without ``--q``/``--g`` the script
lists the candidate table of the canvas (``bragg_peak`` detector, rational
(h, k) attribution in the 1x1 basis, radius rings), writes
``q_candidates.json`` and exits with a non-zero code.

Usage:
    MPLCONFIGDIR=<writable> <venv>/bin/python <this script> CORRECTED.csv \\
        -o OUT_DIR --basis-from correction_report.json --lambda-nm 13 \\
        --q 1/3,1/3 --q 2/3,-1/3 --q 1/3,-2/3
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bpl_figures as fig  # noqa: E402
import bpl_lib as bpl  # noqa: E402

LOG: list[str] = []
CANDIDATE_PRINT_MAX = 60


def emit(text=""):
    """Append one line to the log and echo it (the log is the stdout record)."""
    LOG.append(text)
    print(text, flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="stm_bragg_pair_lockin.py",
        description=(
            "local field and real-space reconstruction for arbitrary Bragg points "
            "(point -> field -> real space; statistics are not part of this skill)"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="corrected topography CSV (square matrix)")
    parser.add_argument(
        "-o",
        "--outdir",
        default=None,
        help="output directory (default: the input directory)",
    )
    parser.add_argument(
        "--basis-from",
        default=None,
        help="correction report JSON (lawler-fujita / affine) giving the 1x1 reciprocal basis",
    )
    parser.add_argument(
        "--basis-px",
        default=None,
        help="explicit basis 'b1x,b1y;b2x,b2y' as signed fftshift offsets in px",
    )
    parser.add_argument(
        "-L",
        "--size-nm",
        type=float,
        default=None,
        help="field of view of the canvas in nm",
    )
    parser.add_argument(
        "--size-nm-from-log",
        default=None,
        help="read the field of view from a correction log",
    )
    parser.add_argument(
        "--lambda-nm",
        type=float,
        default=None,
        help="Gaussian window width in nm (REQUIRED, no default, except with --replot where it is read from the report; this is the window of this skill, not the lawler-fujita lambda)",
    )
    parser.add_argument(
        "--replot",
        action="store_true",
        help="redraw the figures of an existing run from its report.json + h5 + the CSV; no demodulation, no h5, report.json is not rewritten, lambda/basis/FOV come from the report (failures exit 5)",
    )
    parser.add_argument(
        "--lambda-nm-label",
        action="append",
        default=[],
        metavar="LABEL=VALUE",
        help="per-point lambda override (repeatable); the report records the lambda of every point",
    )
    parser.add_argument(
        "--q",
        action="append",
        default=[],
        metavar="H,K",
        help="Bragg point as fractions or decimals, q = h b1 + k b2 (repeatable)",
    )
    parser.add_argument(
        "--g",
        action="append",
        default=[],
        metavar="N:I,J",
        help="Bragg point as (i b1 + j b2) / N (repeatable)",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        metavar="NAME",
        help="label of a point, in the order the points are given (repeatable)",
    )
    parser.add_argument(
        "--list-candidates",
        action="store_true",
        help="also write the candidate table of the canvas when the points are given",
    )
    parser.add_argument(
        "--denominator-max",
        type=int,
        default=bpl.DEFAULT_DENOMINATOR_MAX,
        help="denominator cap of the rational (h, k) attribution of the candidate table",
    )
    parser.add_argument(
        "--ring-tol",
        type=float,
        default=bpl.DEFAULT_RING_TOL,
        help="relative radius tolerance of the candidate rings",
    )
    parser.add_argument(
        "--amplitude-fraction",
        type=float,
        default=bpl.DEFAULT_AMPLITUDE_FRACTION,
        help="mask threshold as a fraction of the median amplitude",
    )
    parser.add_argument(
        "--delimiter", default=bpl.DEFAULT_DELIMITER, help="delimiter of the input CSV"
    )
    parser.add_argument(
        "--steps-only",
        action="append",
        default=[],
        metavar="LABEL",
        help="write the step figures only for this label (repeatable)",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="write the h5 products and the report only, no PNG",
    )
    parser.add_argument(
        "--stm-lib",
        default=bpl.STM_LIB_DEFAULT,
        help="STM_DataProcessing src directory (bragg_peak detector of the candidate table)",
    )
    return parser.parse_args(argv)


def candidate_lines(table, print_max=CANDIDATE_PRINT_MAX):
    """The candidate table as log lines (one row each, capped for the log)."""
    lines = []
    rows = table["candidates"]
    shown = sorted(rows, key=lambda row: (row["ring_id"], row["q_nm_inv_abs"]))[
        :print_max
    ]
    for row in shown:
        partner = row["friedel_partner"]
        cross = row["cross_talk_with_selected"]
        if (
            cross["threshold_rad_per_nm"] is None
            or cross["min_separation_rad_per_nm"] is None
        ):
            cross_text = "cross-talk: no selected point to compare with"
        elif cross["cross_talk"]:
            cross_text = (
                "cross-talk: YES ("
                + ", ".join(
                    f"{item['label']} sign {item['sign']:+d} at "
                    f"{item['separation_rad_per_nm']:.4f} nm^-1"
                    for item in cross["violations"]
                )
                + f", below 2 / lambda = {cross['threshold_rad_per_nm']:.4f} nm^-1)"
            )
        else:
            cross_text = (
                f"cross-talk: none (min |dq| = "
                f"{cross['min_separation_rad_per_nm']:.4f} nm^-1 vs 2 / lambda = "
                f"{cross['threshold_rad_per_nm']:.4f} nm^-1)"
            )
        if cross["same_reflection"]:
            cross_text += "; coincides with " + ", ".join(
                f"{item['label']} sign {item['sign']:+d} (same reflection)"
                for item in cross["same_reflection"]
            )
        distance = partner["detected_peak_distance_px"]
        lines.append(
            f"# candidate {row['ring_id']} #{row['peak_index']}: "
            f"(h, k) = ({row['h_k_fraction'][0]}, {row['h_k_fraction'][1]})  "
            f"[{row['writings']['cli_hk']} | {row['writings']['cli_n_times_basis']}]  "
            f"|q| = {row['q_nm_inv_abs']:.4f} nm^-1  |F| = {row['f_amplitude']:.6g}  "
            f"azimuth = {row['azimuth_deg']:.4f} deg  "
            f"Friedel partner |F| = {partner['amplitude']:.6g} "
            f"(asymmetry residual {partner['asymmetry_residual']:.4g}, "
            f"detected {partner['detected_peak_found']}, "
            + (
                f"distance {distance:.3f} px"
                if distance is not None
                else "no partner peak"
            )
            + f")  rational residual = {row['rational_residual_nm_inv']:.6g} nm^-1 "
            f"({row['rational_residual_relative']:.4g} of |q|, "
            f"{'commensurate' if row['commensurate'] else 'non-commensurate'})  "
            f"ring amplitude residual = {row['amplitude_residual_ring']:.4g}  "
            f"{cross_text}"
        )
    if len(rows) > len(shown):
        lines.append(
            f"# ... {len(rows) - len(shown)} more candidate row(s) in the JSON"
        )
    return lines


def candidate_payload(table, selection_note):
    """The JSON payload of the candidate table."""
    return {
        "skill": "bragg-pair-lockin",
        "skill_version": bpl.SKILL_VERSION,
        "table": "candidate Bragg points of the canvas, attributed in the 1x1 basis of the correction report",
        "note": selection_note,
        "ring_naming": "rings are numbered by ascending radius; no k-space high-symmetry label is used",
        "q_units": "rad/nm (2 pi times cycles/nm) in the 1x1 basis of the correction report",
        **table,
    }


_STATE = {"outdir": None}


def fail(message, code=bpl.EXIT_USAGE):
    """Log an error, mirror it on stderr, flush the log, and exit with ``code``."""
    emit(f"# ERROR: {message}")
    print(f"# ERROR: {message}", file=sys.stderr, flush=True)
    if _STATE["outdir"] is not None:
        write_log(_STATE["outdir"])
    raise SystemExit(int(code))


def write_log(outdir, name="bragg_pair_lockin.log"):
    """Write the log of this run."""
    path = Path(outdir) / name
    path.write_text("\n".join(LOG) + "\n", encoding="utf-8")
    return path


def replot_point(entry):
    """The ``Point`` of a report entry, rebuilt from its exact (h, k) fractions."""
    return bpl.Point(
        label=str(entry["label"]),
        h=Fraction(str(entry["h_fraction"])),
        k=Fraction(str(entry["k_fraction"])),
        source=str(entry.get("source", "report.json")),
        lambda_nm=float(entry["lambda_nm"]),
    )


def matches(first, second, tolerance=1e-9):
    """Numeric agreement of a product attribute with the report."""
    first, second = float(first), float(second)
    return abs(first - second) <= tolerance * max(1.0, abs(first), abs(second))


def replot(args, csv_path, outdir):
    """Redraw the figures of an existing run (DESIGN section 3).

    Only the report, the two h5 products of every point and the CSV (for the
    windowless canvas spectrum of step 1 and step 2) are read.  Nothing is
    demodulated, no h5 is written and ``report.json`` is left untouched; the
    ``# replot`` section is appended to the existing log.  A product that is
    missing, or whose shape, lambda, label or route contradicts the report, is a
    hard failure (exit 5) and no PNG is written.
    """
    report_path = Path(outdir) / "report.json"
    if not report_path.is_file():
        fail(f"--replot: {report_path} is missing; nothing to redraw", bpl.EXIT_REPLOT)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("skill") != "bragg-pair-lockin":
        fail(
            f"--replot: {report_path} is not a bragg-pair-lockin report",
            bpl.EXIT_REPLOT,
        )
    if "coherence" not in report:
        fail(
            f"--replot: {report_path} carries no coherence block, so the step 5 "
            "figures cannot be recomputed from it (it was written before step 5 "
            "existed); rerun the analysis instead of --replot",
            bpl.EXIT_REPLOT,
        )
    emit(f"# replot at {bpl.h5io.utc_now_iso()}")
    emit(
        f"# replot: figures of {report_path} are redrawn from the products; no "
        "demodulation, no h5 write, report.json is not touched"
    )
    if args.lambda_nm is not None:
        emit(
            "# WARNING: --lambda-nm is ignored in --replot mode (it comes from report.json)"
        )
    if args.basis_from or args.basis_px:
        emit(
            "# WARNING: --basis-from/--basis-px are ignored in --replot mode (the basis comes from report.json)"
        )
    if args.size_nm is not None or args.size_nm_from_log:
        emit(
            "# WARNING: -L/--size-nm and --size-nm-from-log are ignored in --replot mode (the field of view comes from report.json)"
        )
    if args.q or args.g or args.label:
        emit(
            "# WARNING: --q/--g/--label are ignored in --replot mode (the points come from report.json)"
        )
    if args.list_candidates:
        emit("# WARNING: --list-candidates has no effect in --replot mode")
    report_delimiter = str(report.get("delimiter", bpl.DEFAULT_DELIMITER))
    if args.delimiter != report_delimiter:
        emit(
            f"# WARNING: --delimiter {args.delimiter!r} is ignored in --replot "
            f"mode; report.json was written with {report_delimiter!r}"
        )
    report_fraction = report.get("amplitude_fraction")
    if report_fraction is not None and not matches(
        args.amplitude_fraction, report_fraction
    ):
        emit(
            "# WARNING: --amplitude-fraction is ignored in --replot mode (the "
            "mask of every point comes from its h5 product)"
        )
    if args.no_figures:
        emit(
            "# WARNING: --no-figures was given, so --replot has nothing to draw "
            "(exit 0); report.json stays as it is"
        )
        write_replot_log(outdir)
        return 0

    canvas, _ = bpl.load_topo(csv_path, report_delimiter)
    n_px = int(canvas.shape[0])
    canvas_px = int(report["canvas_px"])
    if n_px != canvas_px:
        fail(
            f"--replot: the CSV is {n_px} x {n_px} px but report.json was written "
            f"for {canvas_px} px",
            bpl.EXIT_REPLOT,
        )
    nm_per_px = float(report["nm_per_px"])
    size_nm = float(report["field_of_view_nm"])
    stem = csv_path.stem
    coherence_block = report["coherence"]
    all_entries = list(report["q"])
    known = [str(entry["label"]) for entry in all_entries]
    for step_label in args.steps_only:
        if step_label not in known:
            fail(
                f"--steps-only {step_label!r}: no point is labelled {step_label!r} "
                f"in report.json (recorded: {', '.join(known)})",
                bpl.EXIT_USAGE,
            )
    selected = [
        entry
        for entry in all_entries
        if not args.steps_only or str(entry["label"]) in args.steps_only
    ]

    # ------------------------------------------------------- validate first
    plans = []
    for entry in selected:
        label = str(entry["label"])
        artifacts = entry["artifacts"]
        field_path = Path(outdir) / Path(artifacts["h5"]).name
        pair_path = Path(outdir) / Path(artifacts["B_h5"]).name
        for kind, path, suffix in (
            ("field product", field_path, f"_{label}.h5"),
            ("pair product", pair_path, f"_{label}_B.h5"),
        ):
            if not path.is_file():
                fail(
                    f"--replot: the {kind} {path} named by report.json (resolved by "
                    "basename in the output directory) is missing",
                    bpl.EXIT_REPLOT,
                )
            if not path.name.endswith(suffix):
                fail(
                    f"--replot: the {kind} {path.name} does not carry the label "
                    f"{label!r} recorded in report.json",
                    bpl.EXIT_REPLOT,
                )
        with h5py.File(field_path, "r") as handle:
            missing = [
                name
                for name in ("field", "amplitude", "theta", "mask")
                if name not in handle
            ]
            if missing:
                fail(
                    f"--replot: {field_path.name} lacks {missing}",
                    bpl.EXIT_REPLOT,
                )
            shape = tuple(handle["field"].shape)
            if shape != (canvas_px, canvas_px):
                fail(
                    f"--replot: {field_path.name} has shape {shape} but report.json "
                    f"says {canvas_px} x {canvas_px} px",
                    bpl.EXIT_REPLOT,
                )
            theta = handle["theta"][:]
            amplitude = handle["amplitude"][:]
            mask = handle["mask"][:]
        with h5py.File(pair_path, "r") as handle:
            if "B" not in handle:
                fail(f"--replot: {pair_path.name} lacks the B dataset", bpl.EXIT_REPLOT)
            shape = tuple(handle["B"].shape)
            if shape != (canvas_px, canvas_px):
                fail(
                    f"--replot: {pair_path.name} has shape {shape} but report.json "
                    f"says {canvas_px} x {canvas_px} px",
                    bpl.EXIT_REPLOT,
                )
            attributes = dict(handle.attrs)
            B = handle["B"][:]
        for key, recorded in (
            ("label", label),
            ("route", str(entry["route"])),
        ):
            if str(attributes.get(key)) != recorded:
                fail(
                    f"--replot: {pair_path.name} attribute {key} is "
                    f"{attributes.get(key)!r} but report.json says {recorded!r}",
                    bpl.EXIT_REPLOT,
                )
        for key, recorded in (
            ("lambda_nm", float(entry["lambda_nm"])),
            ("q_nm_inv_abs", float(entry["q_nm_inv_abs"])),
        ):
            if not matches(attributes.get(key, float("nan")), recorded):
                fail(
                    f"--replot: {pair_path.name} attribute {key} is "
                    f"{attributes.get(key)!r} but report.json says {recorded!r}",
                    bpl.EXIT_REPLOT,
                )
        plans.append(
            {
                "entry": entry,
                "point": replot_point(entry),
                "B": B,
                "theta": theta,
                "amplitude": amplitude,
                "valid": np.asarray(mask, dtype=float) > 0.5,
            }
        )

    # ------------------------------------------------------------- draw
    fig.setup_style()
    magnitude = np.abs(np.fft.fftshift(np.fft.fft2(np.nan_to_num(canvas, nan=0.0))))
    others = [
        {"label": str(entry["label"]), "q_px": [float(v) for v in entry["q_px"]]}
        for entry in all_entries
    ]
    written = []
    for plan in plans:
        entry = plan["entry"]
        point = plan["point"]
        label = point.label
        valid, theta, amplitude = plan["valid"], plan["theta"], plan["amplitude"]
        B = plan["B"]
        artifacts = fig.point_previews(
            outdir, stem, label, valid, amplitude, np.degrees(theta), valid
        )
        b_max = float(np.percentile(np.abs(B[valid]), 99.5))
        artifacts["B_png"] = fig.bare_map(
            outdir / f"{stem}_{label}_B.png",
            np.where(valid, B, np.nan),
            cmap="viridis",
            vmin=-b_max,
            vmax=b_max,
        )
        written.extend(str(path) for path in artifacts.values())
        steps = []
        if not args.steps_only or label in args.steps_only:
            with h5py.File(
                Path(outdir) / Path(entry["artifacts"]["h5"]).name, "r"
            ) as handle:
                field = handle["field"][:]
            steps = fig.step_figures(
                outdir,
                point,
                entry,
                canvas,
                magnitude,
                nm_per_px,
                field,
                amplitude,
                theta,
                valid,
                others,
            )
            record = entry.get("coherence", {})
            coherence = bpl.phase_coherence(
                theta,
                valid,
                nm_per_px=nm_per_px,
                lambda_nm=point.lambda_nm,
                max_lag_nm=coherence_block["max_lag_nm"],
                fit_range_nm=coherence_block["fit_range_nm"],
                ramp_cut_nm=record.get("ramp_cut_nm"),
                ramp_fit_max_nm=record.get("ramp_fit_max_nm"),
            )
            steps.append(
                fig.step5_phase_coherence(outdir, point, entry, coherence, nm_per_px)
            )
            written.extend(str(path) for path in steps)
        emit(
            f"# replot {label}: {(entry['writings'])['text']}   route = {entry['route']}"
        )
        for path in list(artifacts.values()) + steps:
            emit(f"# replot {label}: wrote {path}")

    if not args.no_figures and all_entries:
        summary_entries = []
        for entry in all_entries:
            label = str(entry["label"])
            pair_path = Path(outdir) / Path(entry["artifacts"]["B_h5"]).name
            with h5py.File(pair_path, "r") as handle:
                summary_entries.append(
                    {
                        "label": label,
                        "B": handle["B"][:],
                        "caption": fig.caption_for(replot_point(entry), entry),
                        "B_rms": float(entry["reconstruction"]["B_rms"]),
                    }
                )
        written.append(str(fig.b_summary(outdir, summary_entries, size_nm)))
    emit(f"# replot: {len(plans)} point(s), {len(written)} file(s) redrawn")
    write_replot_log(outdir)
    return 0


def write_replot_log(outdir, name="bragg_pair_lockin.log"):
    """Append the ``# replot`` section to the existing log of the run."""
    path = Path(outdir) / name
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(LOG) + "\n")
    return path


def main(argv=None):
    args = parse_args(argv)
    csv_path = Path(args.input)
    if not csv_path.is_file():
        fail(f"{csv_path}: no such file", bpl.EXIT_INPUT)
    outdir = Path(args.outdir) if args.outdir else csv_path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    _STATE["outdir"] = outdir
    stem = csv_path.stem

    if args.replot:
        return replot(args, csv_path, outdir)
    if args.lambda_nm is None:
        fail(
            "--lambda-nm is required (no default); it is only optional with "
            "--replot, where it is read from report.json",
            bpl.EXIT_USAGE,
        )
    if (args.basis_from is None) == (args.basis_px is None):
        fail(
            "exactly one basis source is required: --basis-from REPORT.json or "
            "--basis-px 'b1x,b1y;b2x,b2y'",
            bpl.EXIT_USAGE,
        )
    if args.denominator_max < 1:
        fail("--denominator-max must be a positive integer", bpl.EXIT_USAGE)
    if not (0.0 <= args.ring_tol < 1.0):
        fail("--ring-tol must be in [0, 1)", bpl.EXIT_USAGE)

    canvas, nan_fraction = bpl.load_topo(csv_path, args.delimiter)
    n_px = int(canvas.shape[0])
    nan_region = ~np.isfinite(canvas)

    lq = bpl.lq
    basis = (
        lq.basis_from_report(args.basis_from)
        if args.basis_from
        else lq.basis_from_px(args.basis_px)
    )
    size_nm, fov_source = bpl.resolve_field_of_view(
        args.size_nm, args.size_nm_from_log, basis, n_px
    )
    nm_per_px = size_nm / float(n_px)
    resolved = lq.resolve_basis(basis, n_px, nm_per_px)
    b1_rad_px = np.asarray(resolved["b1_rad_px"], dtype=float)
    b2_rad_px = np.asarray(resolved["b2_rad_px"], dtype=float)
    basis_nm_per_px = float(resolved["basis_nm_per_px"])
    nm_per_px_ratio = basis_nm_per_px / nm_per_px
    warnings = []

    emit(f"# bragg-pair-lockin v{bpl.SKILL_VERSION}")
    emit(
        "# layer: Bragg point -> local field -> real-space B(r); no statistics, no atlas"
    )
    emit(f"# input: {csv_path}")
    emit(
        f"# engine: {bpl.lq.__file__} (the sibling skill local-q-map, imported in-process)"
    )
    emit(
        f"# canvas {n_px} x {n_px} px, field of view {size_nm:.4f} nm "
        f"({nm_per_px:.8f} nm/px), NaN {100.0 * nan_fraction:.2f} %"
    )
    emit(f"# field of view source: {fov_source}")
    emit(
        f"# basis source: {resolved['kind']}"
        + (f" ({resolved['path']})" if resolved.get("path") else "")
    )
    emit(f"# basis keys used: {', '.join(resolved['detected_keys'])}")
    for note in resolved.get("notes", []):
        emit(f"# basis note: {note}")
    emit(
        f"# basis b1 = ({resolved['b1_nm_inv'][0]:.6f}, {resolved['b1_nm_inv'][1]:.6f}) "
        f"nm^-1 at {resolved['b1_angle_deg']:.4f} deg, b2 = "
        f"({resolved['b2_nm_inv'][0]:.6f}, {resolved['b2_nm_inv'][1]:.6f}) nm^-1 at "
        f"{resolved['b2_angle_deg']:.4f} deg, angle between them "
        f"{resolved['angle_between_deg']:.4f} deg, |b1| = "
        f"{resolved['abs_b1_nm_inv']:.6f} nm^-1"
    )
    emit(
        f"# basis nm/px used for the radial conversion: {basis_nm_per_px:.8f} "
        f"({resolved['basis_conversion']})"
    )
    if abs(nm_per_px_ratio - 1.0) > lq.NM_PER_PX_REL_TOLERANCE:
        warnings.append(
            f"nm/px mismatch: the canvas gives L / N = {nm_per_px:.8f} nm/px "
            f"({fov_source}) but the basis source converts with {basis_nm_per_px:.8f} "
            f"nm/px ({resolved['basis_conversion']}): ratio {nm_per_px_ratio:.6f}. "
            "The reported q_nm_inv and lambda_px follow the canvas value, the basis "
            "vectors follow the report"
        )
    emit(
        f"# window lambda = {args.lambda_nm:.4f} nm = "
        f"{lq.lambda_px(args.lambda_nm, nm_per_px):.4f} px (this skill's local Fourier "
        "window; the lawler-fujita lambda of the report is NOT this number)"
    )
    if float(args.lambda_nm) > size_nm / 4.0:
        warnings.append(
            f"lambda = {args.lambda_nm:.4f} nm exceeds L / 4 = {size_nm / 4.0:.4f} nm: "
            "the window is not small against the field of view, the periodic FFT "
            "pollutes most of the canvas"
        )
    for text in resolved.get("warnings", []):
        warnings.append(str(text))

    if not (args.q or args.g):
        emit("")
        emit(
            "# no --q/--g: the point set must be given explicitly; listing the "
            "candidate table of the canvas and exiting with a non-zero code"
        )
        table = bpl.candidate_table(
            canvas,
            size_nm,
            resolved,
            denominator_max=args.denominator_max,
            ring_tol=args.ring_tol,
            lambda_nm=args.lambda_nm,
            selected=[],
            stm_lib=args.stm_lib,
        )
        emit(
            f"# detector: {table['detector']}, {table['n_peaks']} peak(s) from "
            f"{table['n_candidates_before_cap']} candidate(s), "
            f"{len(table['rings'])} ring(s) at ring_tol {table['ring_tol']:.4g}, "
            f"denominator cap {table['denominator_max']}"
        )
        for ring in table["rings"]:
            emit(
                f"# {ring['ring_id']}: {ring['n_members']} member(s), radius "
                f"{ring['radius_nm_inv_mean']:.4f} nm^-1 (spread "
                f"{100.0 * ring['radius_spread_relative']:.4f} %), |F| max "
                f"{ring['f_amplitude_max']:.6g}, amplitude residual max "
                f"{ring['amplitude_residual_max']:.4g}"
            )
        for line in candidate_lines(table):
            emit(line)
        candidates_path = outdir / "q_candidates.json"
        bpl.dump_json(
            candidates_path,
            candidate_payload(
                table,
                "written because no --q/--g was given; pick points from the "
                "candidates (or give any other (h, k) by hand) and rerun",
            ),
        )
        emit(f"# wrote {candidates_path}  ({len(table['candidates'])} row(s))")
        if not table["candidates"]:
            emit(
                "# ERROR: the detector found no Bragg peak, so the candidate table is "
                f"empty (exit {bpl.EXIT_INPUT}): give the points explicitly with --q/--g"
            )
            write_log(outdir)
            return bpl.EXIT_INPUT
        emit(f"# no point set was selected (exit {bpl.EXIT_USAGE})")
        write_log(outdir)
        return bpl.EXIT_USAGE

    points, notices = bpl.build_points(
        args.q, args.g, args.label, args.lambda_nm, args.lambda_nm_label
    )
    known = [point.label for point in points]
    for step_label in args.steps_only:
        if step_label not in known:
            fail(
                f"--steps-only {step_label!r}: no point is labelled {step_label!r} "
                f"(selected: {', '.join(known)})",
                bpl.EXIT_USAGE,
            )
    q_rad_px_by_point = {
        point.label: float(point.h) * b1_rad_px + float(point.k) * b2_rad_px
        for point in points
    }
    q_nm_inv_by_point = {
        label: np.asarray(vector, dtype=float) / nm_per_px
        for label, vector in q_rad_px_by_point.items()
    }
    q_px_by_point = {
        label: lq.rad_px_to_px_offsets(vector, n_px)
        for label, vector in q_rad_px_by_point.items()
    }
    # the candidate table lives in the 1x1 basis of the correction report, so its
    # cross-talk column is measured against the same representation of the points
    selected = [(point.label, bpl.q_of(point.h, point.k, resolved)) for point in points]

    emit("")
    emit(
        f"# point set: {len(points)} point(s) -> {len(points)} Friedel pair(s); every "
        "requested q is paired with -q automatically"
    )
    for notice in notices:
        emit(f"# NOTE: {notice}")
    for point in points:
        write = point.writings
        emit(
            f"# {point.label}: (h, k) = ({bpl.fraction_text(point.h)}, "
            f"{bpl.fraction_text(point.k)}) = {write['n_times_basis']} x basis   "
            f"[{write['cli_hk']} | {write['cli_n_times_basis']}]   "
            f"lambda = {point.lambda_nm:.4f} nm"
        )
        if point.phase_degenerate:
            message = (
                f"{point.label} is self-conjugate (2q is a reciprocal lattice vector, "
                "q = 0 included): the Friedel partner is the point itself and the "
                "phase is degenerate; the products are still written "
                "(phase_degenerate = true)"
            )
            warnings.append(message)
            emit(f"# WARNING: {message}")
    if not args.no_figures:
        fig.setup_style()

    magnitude = None
    if not args.no_figures:
        magnitude = np.abs(np.fft.fftshift(np.fft.fft2(np.nan_to_num(canvas, nan=0.0))))

    coherence_block = bpl.coherence_config(size_nm, args.lambda_nm, nm_per_px)
    emit(
        f"# step 5 (phase coherence) is always on: max_lag = L / sqrt(2) = "
        f"{coherence_block['max_lag_nm']:.4f} nm, fit range = (2/3) max_lag = "
        f"{coherence_block['fit_range_nm']:.4f} nm, border erosion = 0.65 lambda = "
        f"{coherence_block['border_erosion_px']:.4f} px"
    )
    emit(
        f"# step 5: the ramp cut is capped at min(L/2, 2 * min(inset_x, inset_y)) "
        f"* nm_per_px, a lag is valid from a pair count of "
        f"{coherence_block['pair_floor']} on, and the strain uses A = "
        f"{coherence_block['A_per_strain_angstrom_inv']} Angstrom^-1 (divisor "
        f"{coherence_block['strain_divisor_per_nm']:.6f} per nm)"
    )

    entries = []
    b_arrays = {}
    written = []
    for index, point in enumerate(points):
        label = point.label
        q_rad_px = q_rad_px_by_point[label]
        q_nm_inv = q_nm_inv_by_point[label]
        q_px = q_px_by_point[label]
        q_abs = float(np.hypot(*q_nm_inv))
        azimuth = float(np.degrees(np.arctan2(q_px[1], q_px[0])) % 360.0)

        field = bpl.local_field(canvas, q_rad_px, point.lambda_nm, nm_per_px)
        amplitude = np.abs(field)
        if not np.any(amplitude > 0.0):
            fail(
                f"{label}: the local field is identically zero on this canvas; there "
                "is no field to reconstruct",
                bpl.EXIT_DEGENERATE,
            )
        valid, threshold = lq.valid_mask(amplitude, nan_region, args.amplitude_fraction)
        theta = np.angle(field)
        coherence = bpl.phase_coherence(
            theta,
            valid,
            nm_per_px=nm_per_px,
            lambda_nm=point.lambda_nm,
            max_lag_nm=coherence_block["max_lag_nm"],
            fit_range_nm=coherence_block["fit_range_nm"],
        )

        others = [
            (other.label, q_nm_inv_by_point[other.label])
            for other in points
            if other.label != label
        ]
        record = bpl.criteria(field, valid, q_rad_px, q_nm_inv, point.lambda_nm, others)
        analytic = bpl.analytic_map(field, q_rad_px)
        two_sided, imaginary_max = bpl.two_sided_map(
            canvas, q_rad_px, point.lambda_nm, nm_per_px
        )
        check = bpl.cross_check(analytic, two_sided, valid)
        B = analytic if record["route"] == bpl.ROUTE_ANALYTIC else two_sided
        summary = bpl.statistical_summary(field, amplitude, valid)

        prefix = f"{stem}_{label}"
        artifacts = {"h5": outdir / f"{prefix}.h5", "B_h5": outdir / f"{prefix}_B.h5"}
        bpl.write_point_products(
            artifacts["h5"], field=field, amplitude=amplitude, theta=theta, mask=valid
        )
        bpl.write_pair_products(
            artifacts["B_h5"],
            B=B,
            mask=valid,
            attributes={
                "label": label,
                "q_h": bpl.fraction_text(point.h),
                "q_k": bpl.fraction_text(point.k),
                "q_writing_hk": point.writings["hk"],
                "q_writing_n_times_basis": point.writings["n_times_basis"],
                "q_nm_inv_abs": q_abs,
                "lambda_nm": float(point.lambda_nm),
                "route": str(record["route"]),
                "reconstruction": bpl.FORMULA_ANALYTIC
                if record["route"] == bpl.ROUTE_ANALYTIC
                else bpl.FORMULA_TWO_SIDED,
                "phase_degenerate": int(point.phase_degenerate),
                "n_valid_pixels": int(np.count_nonzero(valid)),
            },
        )
        if not args.no_figures:
            artifacts.update(
                fig.point_previews(
                    outdir, stem, label, valid, amplitude, np.degrees(theta), valid
                )
            )
            b_max = float(np.percentile(np.abs(B[valid]), 99.5))
            artifacts["B_png"] = fig.bare_map(
                outdir / f"{prefix}_B.png",
                np.where(valid, B, np.nan),
                cmap="viridis",
                vmin=-b_max,
                vmax=b_max,
            )
        written.extend(str(path) for path in artifacts.values())

        emit("")
        emit(
            f"# {label}: (h, k) = ({bpl.fraction_text(point.h)}, "
            f"{bpl.fraction_text(point.k)}) = {point.writings['n_times_basis']} x basis   "
            f"[{point.writings['cli_hk']} | {point.writings['cli_n_times_basis']}]   "
            f"source {point.source}   {point.writings['text']}"
        )
        emit(
            f"# {label}: q_px ({q_px[0]:.4f}, {q_px[1]:.4f}), |q| = {q_abs:.6f} nm^-1 = "
            f"{q_abs / (2.0 * np.pi):.6f} cycles/nm, azimuth {azimuth:.4f} deg, "
            f"lambda = {point.lambda_nm:.4f} nm = "
            f"{lq.lambda_px(point.lambda_nm, nm_per_px):.4f} px"
        )
        emit(
            f"# {label}: criteria: 2 lambda^2 |q|^2 = {record['two_lambda2_q2']:.6g} "
            f"(Friedel suppression {record['friedel_suppression']:.6g}, threshold "
            f"{bpl.FRIEDEL_EXPONENT_MIN:.6g}, ok={record['friedel_ok']}); smallest |dq| "
            f"= {bpl.number_text(record['crosstalk_min_separation_rad_per_nm'])} nm^-1 vs "
            f"2 / lambda = {record['crosstalk_threshold_rad_per_nm']:.6g} nm^-1 "
            f"(ok={record['crosstalk_ok']}); envelope change per carrier cycle = "
            f"{bpl.number_text(record['slow_variation_per_cycle'])} vs max "
            f"{record['slow_variation_max']:.4g} (ok={record['slow_ok']})"
        )
        emit(f"# {label}: route = {record['route']} -- {record['route_reason']}")
        emit(
            f"# {label}: route cross-check (analytic vs two-sided): correlation "
            f"{check['correlation']:.6f}, rms ratio two_sided/analytic "
            f"{check['rms_ratio_two_sided_over_analytic']:.6f}, max |difference| "
            f"{check['max_abs_difference']:.4g}"
        )
        emit(
            f"# {label}: amplitude median {summary['amplitude']['median']:.6g}, FWHM "
            f"{summary['amplitude']['fwhm']:.6g}; theta circular median "
            f"{summary['theta']['circular_median_deg']:.4f} deg, R = "
            f"{summary['theta']['resultant_R']:.6f}; mask coverage "
            f"{100.0 * summary['mask_coverage_fraction']:.3f} % "
            f"(nan fraction {100.0 * nan_fraction:.3f} %)"
        )
        emit(
            f"# {label}: B(r) rms {float(np.std(B[valid])):.6g}, max |B| "
            f"{float(np.max(np.abs(B[valid]))):.6g}; two-sided imaginary maximum "
            f"{imaginary_max:.4g}"
        )
        coherence_record = coherence.record
        emit(
            f"# {label}: coherence: {coherence_record['n_valid_px']} valid px "
            f"({100.0 * coherence_record['valid_fraction']:.3f} %), border "
            f"{coherence_record['border_erosion_px']:.4f} px; |C| at "
            + ", ".join(
                f"{fraction} max_lag = {value:.6g}"
                for fraction, value in coherence_record["anchors"].items()
            )
        )
        emit(
            f"# {label}: coherence: xi = {coherence_record['xi_nm']:.6g} nm over "
            f"{int(np.count_nonzero((coherence.lag_nm > 0) & (coherence.lag_nm <= coherence.fit_range_nm)))} "
            f"lag bin(s) up to {coherence.fit_range_nm:.4f} nm (no sample-count "
            f"filter); profile last finite lag = "
            f"{coherence_record['profile_last_lag_nm']:.6g} nm with "
            f"{coherence_record['profile_last_pairs']:.6g} valid pixel pair(s)"
        )
        if coherence_record["ramp_cut_truncated"]:
            warnings.append(
                f"{label}: the arg C ramp cut is truncated to "
                f"{coherence_record['ramp_cut_nm']:.4f} nm (cap = min(L/2, "
                f"2 * min(inset_x, inset_y)) * nm_per_px = "
                f"{coherence_record['ramp_cap_nm']:.4f} nm) instead of fit_range = "
                f"{coherence_block['fit_range_nm']:.4f} nm; beyond twice the mask "
                f"inset (inset_x = {coherence_record['inset_x_px']} px, inset_y = "
                f"{coherence_record['inset_y_px']} px, wrap free up to "
                f"{coherence_record['wrap_free_lag_nm']:.4f} nm) the circular "
                "autocorrelation folds the canvas into arg C"
            )
            emit(f"# WARNING: {warnings[-1]}")
        for axis in ("x", "y"):
            ramp = coherence_record["ramp"][axis]
            emit(
                f"# {label}: coherence ramp {axis}: "
                f"{ramp['slope_rad_per_nm']:+.9e} rad/nm = "
                f"{ramp['slope_per_angstrom']:+.9e} Angstrom^-1 -> strain_equivalent "
                f"{ramp['strain_equivalent']:.9e} (cut "
                f"{ramp['cut_nm']:.4f} nm, truncated "
                f"{ramp['cut_truncated']}; exact divisor "
                f"{bpl.STRAIN_DIVISOR_PER_NM:.6f} per nm, A = "
                f"{bpl.STRAIN_A_PER_ANGSTROM_INV} Angstrom^-1)"
            )

        entry = {
            "label": label,
            "pair_index": index,
            "h": float(point.h),
            "k": float(point.k),
            "h_fraction": bpl.fraction_text(point.h),
            "k_fraction": bpl.fraction_text(point.k),
            "writings": point.writings,
            "source": point.source,
            "lambda_nm": float(point.lambda_nm),
            "lambda_px": float(lq.lambda_px(point.lambda_nm, nm_per_px)),
            "q_px": [float(value) for value in q_px],
            "q_rad_px": [float(value) for value in q_rad_px],
            "q_nm_inv": [float(value) for value in q_nm_inv],
            "q_nm_inv_abs": q_abs,
            "q_cycles_per_nm": q_abs / (2.0 * np.pi),
            "azimuth_deg": azimuth,
            "amplitude_fraction": float(args.amplitude_fraction),
            "amplitude_threshold": float(threshold),
            "amplitude": summary["amplitude"],
            "theta": summary["theta"],
            "mask_coverage_fraction": summary["mask_coverage_fraction"],
            "n_valid_pixels": int(np.count_nonzero(valid)),
            "phase_degenerate": bool(point.phase_degenerate),
            "self_conjugate": bool(point.phase_degenerate),
            "route": str(record["route"]),
            "criteria": record,
            "reconstruction": {
                "formula": bpl.FORMULA_ANALYTIC
                if record["route"] == bpl.ROUTE_ANALYTIC
                else bpl.FORMULA_TWO_SIDED,
                "route": str(record["route"]),
                "two_sided_imaginary_max": float(imaginary_max),
                "B_rms": float(np.std(B[valid])),
                "B_max_abs": float(np.max(np.abs(B[valid]))),
            },
            "route_check": check,
            "coherence": coherence_record,
            "artifacts": {key: str(path) for key, path in artifacts.items()},
            "steps": [],
        }
        if not args.no_figures and (not args.steps_only or label in args.steps_only):
            steps = fig.step_figures(
                outdir,
                point,
                entry,
                canvas,
                magnitude,
                nm_per_px,
                field,
                amplitude,
                theta,
                valid,
                [
                    {
                        "label": other.label,
                        "q_px": [float(value) for value in q_px_by_point[other.label]],
                    }
                    for other in points
                ],
            )
            steps.append(
                fig.step5_phase_coherence(outdir, point, entry, coherence, nm_per_px)
            )
            entry["steps"] = [str(path) for path in steps]
            written.extend(str(path) for path in steps)
        entries.append(entry)
        b_arrays[label] = B
        for key, path in entry["artifacts"].items():
            emit(f"# {label}: {key} -> {path}")
        for path in entry["steps"]:
            emit(f"# {label}: step figure -> {path}")

    pairs = []
    for entry in entries:
        pairs.append(
            {
                "pair_index": entry["pair_index"],
                "label": entry["label"],
                "partner_label": f"-{entry['label']}",
                "h": entry["h"],
                "k": entry["k"],
                "writings": entry["writings"],
                "q_nm_inv_abs": entry["q_nm_inv_abs"],
                "lambda_nm": entry["lambda_nm"],
                "route": entry["route"],
                "phase_degenerate": entry["phase_degenerate"],
                "aliases": next(
                    (
                        point.aliases
                        for point in points
                        if point.label == entry["label"]
                    ),
                    [],
                ),
                "reconstruction": entry["reconstruction"]["formula"],
                "B": {
                    "rms": entry["reconstruction"]["B_rms"],
                    "max_abs": entry["reconstruction"]["B_max_abs"],
                },
                "route_check": entry["route_check"],
                "artifacts": {
                    "h5": entry["artifacts"].get("B_h5"),
                    "png": entry["artifacts"].get("B_png"),
                },
            }
        )

    if not args.no_figures and entries:
        caption_by_label = {
            point.label: fig.caption_for(point, entry)
            for point, entry in zip(points, entries, strict=True)
        }
        written.append(
            str(
                fig.b_summary(
                    outdir,
                    [
                        {
                            "label": entry["label"],
                            "B": b_arrays[entry["label"]],
                            "caption": caption_by_label[entry["label"]],
                            "B_rms": entry["reconstruction"]["B_rms"],
                        }
                        for entry in entries
                    ],
                    size_nm,
                )
            )
        )

    candidates_info = None
    if args.list_candidates:
        table = bpl.candidate_table(
            canvas,
            size_nm,
            resolved,
            denominator_max=args.denominator_max,
            ring_tol=args.ring_tol,
            lambda_nm=args.lambda_nm,
            selected=selected,
            stm_lib=args.stm_lib,
        )
        path = bpl.dump_json(
            outdir / "q_candidates.json",
            candidate_payload(
                table,
                "written because --list-candidates was given; the cross-talk column "
                "is measured against the selected points",
            ),
        )
        emit("")
        emit(f"# candidate table ({len(table['candidates'])} row(s)) -> {path}")
        for line in candidate_lines(table):
            emit(line)
        candidates_info = {
            "path": str(path),
            "n_candidates": len(table["candidates"]),
            "n_rings": len(table["rings"]),
            "denominator_max": int(args.denominator_max),
            "ring_tol": float(args.ring_tol),
        }
        written.append(str(path))

    report = {
        "skill": "bragg-pair-lockin",
        "skill_version": bpl.SKILL_VERSION,
        "layer": "Bragg point -> local field -> real-space B(r); statistics belong to phase-analysis",
        "input": str(csv_path),
        "outdir": str(outdir),
        "canvas_px": n_px,
        "field_of_view_nm": float(size_nm),
        "field_of_view_source": fov_source,
        "nm_per_px": float(nm_per_px),
        "nan_fraction": float(nan_fraction),
        "delimiter": args.delimiter,
        "figures": not args.no_figures,
        "steps_only": list(args.steps_only),
        "lambda_nm": float(args.lambda_nm),
        "lambda_nm_note": "window of this skill; --lambda-nm-label overrides it per point",
        "lambda_nm_per_point": {
            entry["label"]: entry["lambda_nm"] for entry in entries
        },
        "amplitude_fraction": float(args.amplitude_fraction),
        "basis_source": {
            "kind": resolved["kind"],
            "path": resolved.get("path"),
            "detected_keys": list(resolved["detected_keys"]),
            "notes": list(resolved.get("notes", [])),
            "basis_conversion": resolved["basis_conversion"],
        },
        "basis": {
            "b1_nm_inv": [float(value) for value in resolved["b1_nm_inv"]],
            "b2_nm_inv": [float(value) for value in resolved["b2_nm_inv"]],
            "b1_rad_px": [float(value) for value in b1_rad_px],
            "b2_rad_px": [float(value) for value in b2_rad_px],
            "b1_px": [float(value) for value in resolved["b1_px"]],
            "b2_px": [float(value) for value in resolved["b2_px"]],
            "abs_b1_nm_inv": float(resolved["abs_b1_nm_inv"]),
            "angle_between_deg": float(resolved["angle_between_deg"]),
            "basis_nm_per_px": basis_nm_per_px,
            "q_definition": "q = h b1 + k b2",
        },
        "coherence": coherence_block,
        "q_units": {
            "q_nm_inv": "rad/nm (2 pi times cycles/nm), the radial convention of the sibling skill local-q-map",
            "q_px": "signed fftshift offset in px on the corrected canvas",
        },
        "criteria_thresholds": {
            "friedel": "analytic requires 2 (lambda |q|)^2 >= -ln(friedel_suppression_max)",
            "friedel_suppression_max": bpl.FRIEDEL_SUPPRESSION_MAX,
            "friedel_exponent_min": bpl.FRIEDEL_EXPONENT_MIN,
            "crosstalk": "analytic requires the smallest |dq| to another point >= 2 / lambda (rad/nm)",
            "slow_variation_max": bpl.SLOW_VARIATION_MAX,
            "slow_variation": "median |psi(r + one carrier period) - psi(r)| / median |psi(r)|",
            "route_check_gate": {
                "correlation_min": 0.9,
                "rms_ratio_two_sided_over_analytic_range": [0.6, 1.4],
                "note": (
                    "the two routes are not identical by construction: the analytic one "
                    "samples the spectrum exactly at q in the demodulated frame, the "
                    "strict one masks the raw-FFT grid around +q and -q; on the r3 points "
                    "of the 20251112 dataset they agree to about 0.9 in rms and 0.95 in "
                    "correlation"
                ),
            },
        },
        "pairing": {
            "rule": "every requested q is paired with its Friedel partner -q; an explicitly given -q is merged into the same pair",
            "n_points": len(entries),
            "n_pairs": len(pairs),
            "n_merged_aliases": sum(len(pair["aliases"]) for pair in pairs),
        },
        "routes": {
            "analytic": [
                e["label"] for e in entries if e["route"] == bpl.ROUTE_ANALYTIC
            ],
            "two_sided": [
                e["label"] for e in entries if e["route"] == bpl.ROUTE_TWO_SIDED
            ],
        },
        "phase_degenerate_labels": [
            e["label"] for e in entries if e["phase_degenerate"]
        ],
        "q": entries,
        "pairs": pairs,
        "candidates": candidates_info,
        "warnings": warnings,
        "log": str(outdir / "bragg_pair_lockin.log"),
        "report": str(outdir / "report.json"),
        "written": written,
    }
    report_path = bpl.dump_json(outdir / "report.json", report)
    emit("")
    emit(
        f"# report -> {report_path}  ({len(entries)} point(s), {len(pairs)} pair(s), "
        f"routes: analytic {report['routes']['analytic']}, two_sided "
        f"{report['routes']['two_sided']})"
    )
    log_path = write_log(outdir)
    emit(f"# log -> {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
