"""Command line entry point for the multi-process Re chi0(q, 0) driver.

The BLAS thread pools are pinned to a single thread *before* NumPy is imported:
the pools are created while NumPy (and its BLAS) is loaded, so a later
assignment has no effect, and with one thread per process the worker count is
the only concurrency knob (otherwise every worker would spawn its own pool and
the processes would fight for the same cores).

Usage
-----
    python scripts/run_lindhard_re_chi_parallel.py \
        --model-dir /path/to/model --seedname C6LiC6_0 --nk 64 \
        --orbitals 48,49,50,51,52 --workers 4 \
        --output out/c6lic6_li_rechi0_64pts.h5 \
        --checkpoint-dir out/ckpt

Add ``--mirror`` to evaluate only the ``iq1 in [0, nk // 2]`` rows and rebuild
the rest from ``chi0(q) = chi0(-q)``, ``--dry-run`` to print the slice plan and
the memory estimate without starting anything, and ``--resume`` to continue from
the complete slices in ``--checkpoint-dir``.
"""

import os

# Pin the BLAS thread pools of every numerical library NumPy may be linked
# against, before the first NumPy import of this process.  ``setdefault`` keeps
# a value the caller exported on purpose.
_BLAS_THREAD_ENV_VARS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_BLAS_THREAD_DECISIONS = [
    name for name in _BLAS_THREAD_ENV_VARS if os.environ.setdefault(name, "1") == "1"
]
_BLAS_THREAD_OVERRIDDEN = [
    name for name in _BLAS_THREAD_ENV_VARS if name not in _BLAS_THREAD_DECISIONS
]

import argparse  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from stm_data_processing.dft.wannier90.lindhard_re_chi_parallel import (  # noqa: E402
    configure_logging,
    run_parallel,
)

logger = logging.getLogger(__name__)


def default_workers() -> int:
    """``min(cpu_count - 1, 8)``: leave one core for the parent, cap at eight."""
    cpu_count = os.cpu_count() or 2
    return max(1, min(cpu_count - 1, 8))


def parse_orbitals(text: str) -> list[int] | None:
    """``all`` -> None (every Wannier orbital), otherwise ``48,49,50``."""
    stripped = text.strip()
    if stripped.lower() in ("all", "none", ""):
        return None
    try:
        orbitals = [int(part) for part in stripped.split(",") if part.strip() != ""]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--orbitals must be 'all' or a comma separated list of integers, got {text!r}"
        ) from exc
    if not orbitals:
        raise argparse.ArgumentTypeError("--orbitals must not be empty")
    return orbitals


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model-dir", required=True, help="folder with <seedname>_hr.dat"
    )
    parser.add_argument("--seedname", required=True, help="Wannier90 seedname")
    parser.add_argument(
        "--nk", type=int, default=256, help="k/q mesh size per direction"
    )
    parser.add_argument("--eta", type=float, default=5e-3, help="broadening in eV")
    parser.add_argument(
        "--temperature", type=float, default=4.2, help="temperature in K"
    )
    parser.add_argument(
        "--mu", type=float, default=0.0, help="chemical potential in eV"
    )
    parser.add_argument(
        "--orbitals",
        default="all",
        help="'all' or the projected orbital indices, e.g. 48,49,50,51,52",
    )
    parser.add_argument(
        "--projection-label",
        default=None,
        help="value of the h5 'projection' attribute (default: full/custom)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"worker processes (default: min(cpu_count - 1, 8) = {default_workers()})",
    )
    mirror = parser.add_mutually_exclusive_group()
    mirror.add_argument(
        "--mirror",
        dest="mirror",
        action="store_true",
        help="evaluate only rows [0, nk//2] and mirror via chi0(q) = chi0(-q)",
    )
    mirror.add_argument(
        "--no-mirror",
        dest="mirror",
        action="store_false",
        help="evaluate every q1 row (default)",
    )
    parser.set_defaults(mirror=False)
    parser.add_argument("--output", default=None, help="output HDF5 path")
    parser.add_argument(
        "--checkpoint-dir", default=None, help="slice checkpoint folder"
    )
    parser.add_argument(
        "--resume", action="store_true", help="dispatch only the missing slices"
    )
    parser.add_argument(
        "--band-block", type=int, default=None, help="band-m block width"
    )
    parser.add_argument(
        "--block-entries", type=int, default=None, help="working-array entry budget"
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=30.0,
        help="aggregated progress period in seconds (<= 0 disables it)",
    )
    parser.add_argument(
        "--start-method",
        default="spawn",
        choices=("spawn", "fork", "forkserver"),
        help="multiprocessing start method (spawn by default: no fork with BLAS threads)",
    )
    parser.add_argument(
        "--log-file", default=None, help="also write the parent log here"
    )
    parser.add_argument(
        "--max-mem-gb",
        type=float,
        default=None,
        help="memory budget for all workers (default: 80%% of MemAvailable)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the slice plan and the memory estimate, then exit 0",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging("INFO", log_file=args.log_file, console=True, worker_id=None)
    logger.info(
        "[run_lindhard_re_chi_parallel] BLAS thread variables pinned before "
        "importing NumPy: %s%s",
        ", ".join(_BLAS_THREAD_DECISIONS) if _BLAS_THREAD_DECISIONS else "none",
        (
            f"; kept the caller value for {', '.join(_BLAS_THREAD_OVERRIDDEN)}"
            if _BLAS_THREAD_OVERRIDDEN
            else ""
        ),
    )

    orbitals = parse_orbitals(args.orbitals)
    workers = args.workers if args.workers is not None else default_workers()
    projection = args.projection_label
    if projection is None:
        projection = "full" if orbitals is None else "custom"

    logger.info(
        "[run_lindhard_re_chi_parallel] model=%s seedname=%s nk=%d eta=%.6e T=%s "
        "mu=%s orbitals=%s workers=%d mirror=%s resume=%s output=%s",
        args.model_dir,
        args.seedname,
        args.nk,
        args.eta,
        args.temperature,
        args.mu,
        "all" if orbitals is None else orbitals,
        workers,
        args.mirror,
        args.resume,
        args.output,
    )

    started = time.perf_counter()
    exit_code = run_parallel(
        args.model_dir,
        args.seedname,
        nk=args.nk,
        eta=args.eta,
        temperature=args.temperature,
        chemical_potential=args.mu,
        orbital_select=orbitals,
        band_block=args.band_block,
        block_entries=args.block_entries,
        n_workers=workers,
        mirror=args.mirror,
        output_path=args.output,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        projection=projection,
        progress_interval_s=args.progress_interval,
        start_method=args.start_method,
        log_file=args.log_file,
        max_mem_gb=args.max_mem_gb,
        dry_run=args.dry_run,
    )
    elapsed = time.perf_counter() - started

    if exit_code != 0:
        logger.error(
            "[run_lindhard_re_chi_parallel] FAILED exit=%d output=%s",
            exit_code,
            args.output,
        )
        print(
            f"SUMMARY status=failed exit={exit_code} nk={args.nk} workers={workers} "
            f"mirror={args.mirror} output={args.output or '-'} "
            f"checkpoint_dir={args.checkpoint_dir or '-'} elapsed_s={elapsed:.1f}",
            flush=True,
        )
        return exit_code

    if args.dry_run:
        return 0

    size = Path(args.output).stat().st_size if args.output else 0
    print(
        f"SUMMARY status=ok nk={args.nk} workers={workers} mirror={args.mirror} "
        f"orbitals={'all' if orbitals is None else ','.join(str(o) for o in orbitals)} "
        f"projection={projection} output={args.output or '-'} bytes={size} "
        f"checkpoint_dir={args.checkpoint_dir or '-'} resume={args.resume} "
        f"elapsed_s={elapsed:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
