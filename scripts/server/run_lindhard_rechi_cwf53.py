"""Static Re chi0(q, 0) driver for the cwf53 model (server package version).

Computes the real part of the bare Lindhard susceptibility at omega = 0 with the
multi-process driver of ``lindhard_re_chi_parallel`` for the 53-Wannier CWF model
of C6LiC6 generated on iop3F_server (num_wann=53).  Data and output paths are
resolved relative to this script's location (package layout: ``../data`` for
input, ``../h5_data`` for output), so the package runs anywhere without editing
paths.

Two projections are provided:

    full : all 53 orbitals        (orbital_select = None)
    li   : Li block, indices 48..52 (orbital_select = [48,49,50,51,52])
           nominal labels 1s/2s/2px/2py/2pz (see the handover document:
           the nominal labels do not equal the true orbital character; the
           "1s" slot is a window state because the physical Li 1s lies below
           the CWF disentangling window).

Runtime tuning
--------------
``band_block=53`` (the full band-m width of the vectorized q-sum) is passed to
the engine explicitly: the wider block runs ~1.45x faster than the default 16
while changing the result only at machine precision (max |diff| ~ 1.6e-15,
verified on the 53-orbital model).  The earlier package version patched the
module-level ``_MAX_BAND_BLOCK`` instead; the constructor argument replaces that
monkeypatch.  The module's own ``_HK_ROW_BLOCK=512`` chunking already avoids the
Apple Accelerate zgemm segfault; on Linux BLAS neither issue applies.

Cost scaling is nk**4 (16.3x per nk doubling, measured): nk=32 full = 48 s on an
M1 Pro, so nk=256 full ~ 37 h / li ~ 15 h with the wider block.  Peak memory
5-8 GB per process; ``--workers`` scales the wall time and multiplies that.

Products (one per projection, primitives-BZ fftshifted grid):

    h5_data/c6lic6_cwf53_rechi0_{nk}pts.h5
    h5_data/c6lic6_cwf53_li_rechi0_{nk}pts.h5

with ``module_type='real_Lindhard'``, ``eta=0.005``, ``nq=nk``,
``chemical_potential=0.0``, ``temperature=4.2``, ``orbital_select`` and
``projection`` attributes, a ``susceptibility`` dataset of shape (nk, nk) and a
``bvecs`` dataset of shape (3, 3) - all written once through the single HDF5
write path of the driver.

Usage (from the package root or anywhere else; paths are self-locating):
    python scripts/run_lindhard_rechi_cwf53.py --proj full --nk 256 --workers 8
    python scripts/run_lindhard_rechi_cwf53.py --proj li --nk 32   # smoke
"""

import argparse
import os
import time
from pathlib import Path

import h5py
import numpy as np

from stm_data_processing.dft.wannier90.lindhard_re_chi_parallel import (
    assemble_from_checkpoints,
    run_parallel,
)
from stm_data_processing.dft.wannier90.mlwf_hamiltonian import MLWFHamiltonian

# Package layout: <pkg>/data/<seedname>_hr.dat and <pkg>/data/<seedname>.wout,
# outputs go to <pkg>/h5_data (created on demand).
_PKG_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = str(_PKG_ROOT / "data")
H5_DIR = str(_PKG_ROOT / "h5_data")

SEEDNAME = "C6LiC6_0"
MODEL = "cwf53"
ETA = 5e-3  # Lorentzian broadening in eV (5 meV)
TEMPERATURE = 4.2  # K, needed for Fermi occupations and the df/dE substitution
CHEMICAL_POTENTIAL = 0.0  # hr.dat energies are Fermi-level referenced
BAND_BLOCK = 53  # explicit engine block width (replaces the module monkeypatch)

PROJECTIONS = {
    "full": None,
    "li": [48, 49, 50, 51, 52],
}

#: Attributes and datasets every product must carry (delivery contract).
REQUIRED_ATTRS = (
    "module_type",
    "eta",
    "nq",
    "chemical_potential",
    "temperature",
    "orbital_select",
    "projection",
)
REQUIRED_DATASETS = ("susceptibility", "bvecs")


def sanity_report(ham) -> None:
    """Print basic model facts and the H(Gamma) eigenvalue range."""
    print(
        f"SANITY num_wann={ham.num_wann} nrpts={len(ham.r_list)} "
        f"bvecs_norm1={float(np.linalg.norm(np.asarray(ham.bvecs)[0])):.4f}",
        flush=True,
    )
    h0 = np.asarray(ham.hk(np.zeros((1, 3))))[0]
    if not np.allclose(h0, h0.conj().T, atol=1e-10):
        print("SANITY WARNING: H(Gamma) not Hermitian", flush=True)
    ev = np.linalg.eigvalsh(h0)
    print(
        f"SANITY H(Gamma) eigen range [{ev.min():.4f}, {ev.max():.4f}] eV "
        f"({len(ev)} bands)",
        flush=True,
    )


def product_path(nk: int, proj: str) -> str:
    """Delivery file name of one projection."""
    tag = "" if proj == "full" else f"_{proj}"
    return f"{H5_DIR}/c6lic6_{MODEL}{tag}_rechi0_{nk}pts.h5"


def default_checkpoint_dir(nk: int, proj: str) -> Path:
    """Per-projection checkpoint folder, so a killed run can resume."""
    tag = "" if proj == "full" else f"_{proj}"
    return Path(H5_DIR) / f"ckpt_{MODEL}{tag}_{nk}pts"


def verify_product(path: str, nk: int, proj: str) -> str:
    """Check the delivered attributes/datasets and return a report line."""
    with h5py.File(path, "r") as handle:
        attrs = dict(handle.attrs)
        keys = sorted(handle.keys())
        shape = handle["susceptibility"].shape
        bvecs_shape = handle["bvecs"].shape
    missing = [name for name in REQUIRED_ATTRS if name not in attrs]
    missing += [name for name in REQUIRED_DATASETS if name not in keys]
    if missing:
        raise AssertionError(f"{path}: missing {missing}")
    if str(attrs["module_type"]) != "real_Lindhard":
        raise AssertionError(f"{path}: module_type={attrs['module_type']!r}")
    if float(attrs["eta"]) != ETA or int(attrs["nq"]) != nk:
        raise AssertionError(f"{path}: eta/nq mismatch")
    if float(attrs["chemical_potential"]) != CHEMICAL_POTENTIAL:
        raise AssertionError(f"{path}: chemical_potential mismatch")
    if float(attrs["temperature"]) != TEMPERATURE:
        raise AssertionError(f"{path}: temperature mismatch")
    if str(attrs["projection"]) != proj:
        raise AssertionError(f"{path}: projection={attrs['projection']!r}")
    if shape != (nk, nk) or bvecs_shape != (3, 3):
        raise AssertionError(f"{path}: shapes {shape} / {bvecs_shape}")
    return (
        f"CHECK h5 attrs={sorted(attrs)} datasets={keys} "
        f"susceptibility={shape} bvecs={bvecs_shape}"
    )


def run_projection(
    nk: int,
    proj: str,
    orbitals,
    *,
    workers: int,
    mirror: bool,
    resume: bool,
    checkpoint_dir: str | None,
) -> int:
    """Compute one projection through the parallel driver and report on it."""
    out = product_path(nk, proj)
    ckpt = Path(checkpoint_dir) if checkpoint_dir else default_checkpoint_dir(nk, proj)
    Path(H5_DIR).mkdir(parents=True, exist_ok=True)

    print(
        f"RUN proj={proj} orbitals={orbitals} nk={nk} eta={ETA:.2e} "
        f"T={TEMPERATURE} K mu={CHEMICAL_POTENTIAL} band_block={BAND_BLOCK} "
        f"workers={workers} mirror={mirror} resume={resume} ckpt={ckpt}",
        flush=True,
    )
    t0 = time.time()
    code = run_parallel(
        DATA_DIR,
        SEEDNAME,
        nk=nk,
        eta=ETA,
        temperature=TEMPERATURE,
        chemical_potential=CHEMICAL_POTENTIAL,
        orbital_select=orbitals,
        band_block=BAND_BLOCK,
        n_workers=workers,
        mirror=mirror,
        output_path=out,
        checkpoint_dir=str(ckpt),
        resume=resume,
        projection=proj,
    )
    t1 = time.time()
    if code != 0:
        print(f"FAILED proj={proj} exit={code} compute_s={t1 - t0:.1f}", flush=True)
        return code

    # The product carries only the total response, so the intraband/interband
    # split is read back from the checkpoints that produced it.
    result, slices = assemble_from_checkpoints(
        ckpt,
        nk=nk,
        mirror=mirror,
        eta=ETA,
        chemical_potential=CHEMICAL_POTENTIAL,
        temperature=TEMPERATURE,
        orbital_select=orbitals,
    )
    data = np.asarray(result["data"])
    intra = np.asarray(result["intraband"])
    inter = np.asarray(result["interband"])
    i, j = np.unravel_index(np.argmax(data), data.shape)
    print(
        f"DONE proj={proj} compute_s={t1 - t0:.1f} slices={len(slices)} "
        f"max={data.max():.6g} min={data.min():.6g} "
        f"peak_at_frac=({i / nk - 0.5:.4f},{j / nk - 0.5:.4f})",
        flush=True,
    )
    print(
        f"SPLIT sum_intra={intra.sum():.6g} sum_inter={inter.sum():.6g} "
        f"intra_min={intra.min():.3g} inter_min={inter.min():.3g}",
        flush=True,
    )
    err = float(np.abs(data - (intra + inter)).max())
    print(f"CHECK max|data-(intra+inter)|={err:.3e}", flush=True)
    print(verify_product(out, nk, proj), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proj", choices=sorted(PROJECTIONS), default=None)
    parser.add_argument("--nk", type=int, default=256)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(os.cpu_count() or 1, 8)),
        help="worker processes (default: min(cpu_count, 8))",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="evaluate only rows [0, nk//2] and mirror via chi0(q) = chi0(-q)",
    )
    parser.add_argument(
        "--resume", action="store_true", help="dispatch only the missing slices"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="checkpoint folder (default: h5_data/ckpt_<model>_<nk>pts)",
    )
    args = parser.parse_args()

    projs = [args.proj] if args.proj else sorted(PROJECTIONS)

    Path(H5_DIR).mkdir(parents=True, exist_ok=True)
    ham = MLWFHamiltonian.from_seedname(DATA_DIR, SEEDNAME)
    sanity_report(ham)

    exit_code = 0
    for proj in projs:
        code = run_projection(
            args.nk,
            proj,
            PROJECTIONS[proj],
            workers=args.workers,
            mirror=args.mirror,
            resume=args.resume,
            checkpoint_dir=args.checkpoint_dir,
        )
        exit_code = exit_code or code
    print(f"ALL_DONE exit={exit_code}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
