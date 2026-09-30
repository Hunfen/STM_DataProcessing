#!/usr/bin/env python
"""Fit a Lawler-Fujita displacement field for a grid topography CSV.

This is the **third and last step** of the ``grid-lf-pipeline`` skill.  It takes
the preprocessed topography CSV produced by the sibling ``grid_topo.py`` (the
plain square matrix that ``lawler-fujita-correction`` accepts) and runs the
``lawler-fujita-correction`` skill's own fit, ``stm_lf_correct.py``, then
exports the transferable displacement-field bundle (``<stem>_transform.json``
+ ``<stem>_transform.h5``) that Skill B re-samples other maps against.

Nested dependency only: this script never re-implements the fit or any
resampling.  It locates ``skills/lawler-fujita-correction/scripts/stm_lf_correct.py``
by a **relative path from the STM_DataProcessing checkout root** and calls it
with the repository ``--stm-lib`` default; it neither reads from nor writes to
anything under ``skills/lawler-fujita-correction`` besides running that one
script (nothing there is modified).

Applying the fitted field to current / dI/dV maps is **not** part of this skill
-- that is Skill B.  This script stops after the bundle is written.

**Fail-closed.**  A failed fit must not leave a half-finished output directory:
if ``stm_lf_correct.py`` exits non-zero, or its ``correction_report.json``
records an identity fallback (no usable 1x1 ring / too-low lock-in coverage, so
no real displacement field was computed), this script deletes the output
directory, prints the reason, and exits non-zero.  (Steps 1--2 of the skill --
missing ``.h5`` / missing ``Z (m)`` column -- are already fail-closed in
``grid_topo.py``, which exits non-zero on those; this script's own inputs are the
already-validated CSV.)

Usage::

    .venv/bin/python <this script> TOPO.csv -L 30 -o OUT/lf
    .venv/bin/python <this script> TOPO.csv -L 30 -o OUT/lf --lambda-nm 3

The ``-o`` value *is* the ``<out>/lf/`` directory of the pipeline layout: every
<stem>_corrected.*, <stem>_lf.*, <stem>_transform.*, correction_report.json and
correction.log is written straight into it.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "grid_lf_mplconfig")
)

#: The Lawler-Fujita fit is nested at a fixed *relative* path from the checkout root.
LF_SCRIPT = (
    Path("skills") / "lawler-fujita-correction" / "scripts" / "stm_lf_correct.py"
)


def repo_root() -> Path:
    import stm_data_processing

    return Path(stm_data_processing.__file__).resolve().parents[2]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "input", help="preprocessed topography CSV (grid_topo.py output)"
    )
    parser.add_argument(
        "-L",
        "--size-nm",
        type=float,
        required=True,
        help="field of view in nm (the LF input L)",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        required=True,
        help="output directory = the <out>/lf/ directory of the pipeline "
        "layout; fit products land straight in it",
    )
    parser.add_argument(
        "--lambda-nm",
        type=float,
        default=3.0,
        help="real-space scale kept by the LF lock-in low-pass (default 3.0, "
        "the grid recipe value; the LF skill's own default of 30 nm is too "
        "wide for a 30 nm field and it warns about that)",
    )
    parser.add_argument(
        "--save-transform",
        default=None,
        metavar="FILE",
        help="transferable bundle JSON to write (default "
        "<outdir>/<topo-stem>_transform.json; the .h5 u-field is written next "
        "to it by stm_lf_correct.py)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    topo_csv = Path(args.input)
    if not topo_csv.is_file():
        raise SystemExit(f"ERROR: {topo_csv} does not exist")

    lf_dir = Path(args.outdir)
    # Snapshot the directory as it was on entry: a failure must only remove what
    # this run wrote, never files that were already there.
    pre_existing = {p.name for p in lf_dir.iterdir()} if lf_dir.is_dir() else set()
    lf_dir.mkdir(parents=True, exist_ok=True)

    root = repo_root()
    lf_script = root / LF_SCRIPT
    if not lf_script.is_file():
        raise SystemExit(
            f"ERROR: nested dependency not found in the STM_DataProcessing "
            f"checkout at {root}: {lf_script}"
        )

    bundle_json = (
        Path(args.save_transform)
        if args.save_transform
        else lf_dir / f"{topo_csv.stem}_transform.json"
    )
    cmd = [
        sys.executable,
        str(lf_script),
        str(topo_csv),
        "-L",
        f"{args.size_nm:g}",
        "-o",
        str(lf_dir),
        "--lambda-nm",
        f"{args.lambda_nm:g}",
        "--save-transform",
        str(bundle_json),
    ]
    print("$ " + " ".join(shlex.quote(part) for part in cmd), flush=True)

    failed_reason = None
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        failed_reason = f"the LF fit script failed: {exc}"
    else:
        report_path = lf_dir / "correction_report.json"
        if not report_path.is_file():
            failed_reason = (
                f"the LF fit script wrote no correction_report.json at {report_path}"
            )
        else:
            try:
                report = json.loads(report_path.read_text())
            except json.JSONDecodeError as exc:
                failed_reason = f"cannot read {report_path}: {exc}"
            else:
                if report.get("fallback"):
                    failed_reason = (
                        f"the LF fit fell back to identity (no usable displacement "
                        f"field): {report.get('fallback_reason')}"
                    )

    if failed_reason is not None:
        print(f"ERROR: {failed_reason}", file=sys.stderr)
        removed = []
        for path in sorted(lf_dir.iterdir()):
            if path.name not in pre_existing:
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
                removed.append(path.name)
        raise SystemExit(
            f"fail-closed: fit failed, removed {len(removed)} file(s) written by "
            f"this run from {lf_dir} (files that were already there are kept), "
            f"exit non-zero"
        )

    print(f"# LF fit OK; bundle: {bundle_json}")
    print(f"# products in: {lf_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
