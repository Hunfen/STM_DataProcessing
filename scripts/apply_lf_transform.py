#!/usr/bin/env python
"""Apply a lawler-fujita displacement field to a matrix: resample only.

The method's own statement of the transfer is one line (SKILL.md 2.1)::

    corrected(r) = T(r + u(r))

``T`` is the target matrix, in the same plain square-number format as the
original data; ``u`` is the in-plane displacement field of a bundle written by
``stm_lf_correct.py --save-transform`` (``u_x``, ``u_y`` in nm on the reference
grid).  So this script does exactly that and nothing else::

    read T -> flipud (put it in u's index frame) -> sample at r + u(r) -> write

It never touches the target's values: no ``subtractMeanPlane``, no mean
removal, no normalisation.  ``stm_lf_apply.py`` does subtract a fitted plane
from the target, because the fit stage did the same to the topo and self-test
14/14b pins "applying the bundle to that same topo reproduces the fit-stage CSV
byte for byte"; for any other map that subtraction deletes the target's own DC
level and tilt (measured: adding +100 pA or a linear ramp to a current map
changes its output by only ~1e-22).  It re-detects no peaks and writes no extra
products: the only output is ``<stem>_corrected.csv``.

The resampling is delegated to the skill's own ``lf_lib.warp_by_field``, so the
canvas growth, pad, spline order and NaN rules are identical to the skill's.

Run from the repository root::

    .venv/bin/python scripts/apply_lf_transform.py TARGET.csv \
        --transform OUT/bundle.json -L 30 -o OUT_TARGET
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

# lf_lib / bragg_peak can pull in matplotlib, which needs a writable config
# directory (same precaution as scripts/sxm_to_csv.py).
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "dsh_mplconfig"),
)

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_SCRIPTS = REPO_ROOT / "skills" / "lawler-fujita-correction" / "scripts"
sys.path.insert(0, str(SKILL_SCRIPTS))

import lf_lib as lf  # noqa: E402  (needs SKILL_SCRIPTS on sys.path)

from stm_data_processing.utils.bragg_peak import load_image  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "input",
        help="target matrix (plain square numbers, same format as the source data)",
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
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    bundle_path = Path(args.transform)
    payload = json.loads(bundle_path.read_text())
    if payload.get("fallback") or not payload.get("u_field_file"):
        print(
            f"ERROR: {bundle_path} carries no displacement field "
            f"(method={payload.get('method')}, fallback={payload.get('fallback')}): "
            "nothing to apply",
            file=sys.stderr,
        )
        return 2

    with h5py.File(payload["u_field_file"], "r") as handle:
        u_nm = np.stack([handle["u_x"][:], handle["u_y"][:]]).astype(float)
        valid = handle["valid"][:]

    src = Path(args.input)
    target = load_image(src)
    if target.ndim != 2 or target.shape[0] != target.shape[1]:
        print(
            f"ERROR: {src} is not a square matrix (shape {target.shape})",
            file=sys.stderr,
        )
        return 2
    n = int(target.shape[0])

    reference_n = int(payload["n_px_reference"])
    reference_fov = float(payload["field_of_view_nm_reference"])
    rescaled = n != reference_n or float(args.size_nm) != reference_fov
    if rescaled:  # u is physical (nm): carry it onto this pixel grid
        u_nm, valid = lf.resample_field(u_nm, valid, reference_fov, n, args.size_nm)

    # scan order -> array grid (u's index frame); a coordinate convention, not a
    # value operation.
    array = np.flipud(target)
    corrected, n_out, half, magnitude = lf.warp_by_field(
        array,
        u_nm,
        args.size_nm,
        valid=valid,
        pad=int(payload["pad"]),
        order=int(payload["order"]),
    )

    outdir = Path(args.outdir) if args.outdir else src.parent
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{src.stem}_corrected.csv"
    np.savetxt(out, corrected, delimiter=",", fmt="%.10e")

    print("# corrected(r) = T(r + u(r))  --  resample only, target values untouched")
    print(f"# input: {src}  ({n} x {n} px, {args.size_nm:g} nm)")
    print(
        f"# u field: {payload['u_field_file']}  (reference {reference_n} px / "
        f"{reference_fov:g} nm, rescaled={rescaled})"
    )
    print(
        f"# target mean {np.nanmean(target):+.6e} -> corrected finite mean "
        f"{np.nanmean(corrected):+.6e}"
    )
    print(
        f"# canvas {n} -> {n_out} px, half = {half}, max|u| = {magnitude:.3f} px, "
        f"NaN {100 * np.isnan(corrected).mean():.2f} %"
    )
    print(f"# written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
