"""Regression checks for the OpenMX parsers (bugs M2, M17, M18).

Bug M2 (PDOS classification)
----------------------------
load_dos_tree only recognized PDOS files inside atomN/ sub-directories and
required a dot before the orbital label, so real OpenMX DosMain flat file
names like <System>.PDOS.Tetrahedron.atom1.s1 were all classified as atom
totals, and multi-digit orbital numbers (e.g. p10) were truncated.

Bug M17 (.Band spin degrees of freedom)
---------------------------------------
The .Band header line 1 holds "nband nspin mu"; the data section is arranged
spin-outer, k-point-inner.  The parser ignored nspin, so spin-polarized data
was read as a single spin with duplicated k-points and a spurious mid-path
distance jump.

Bug M18 (Unit Ang coordinates)
------------------------------
Atoms.SpeciesAndCoordinates with Unit Ang returned positions_frac=None and
read_atomic_positions raised RuntimeError even when the lattice vectors were
available; now positions_frac = positions_ang @ inv(avecs) is used.

All fixtures are synthetic OpenMX-style files written to
tmp_verify/openmx_fixture/ (no real OpenMX output exists in this repo).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

# Keep matplotlib cache warnings out of the regression output.
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

from stm_data_processing.dft.openmx.band import parse_dft_band_data
from stm_data_processing.dft.openmx.dos import load_dos_tree
from stm_data_processing.dft.openmx.parser import OpenMX

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "tmp_verify" / "openmx_fixture"

DOS_CONTENT = "# E DOS IDOS\n-10.0 0.5 1.0\n-9.0 0.4 1.4\n"
PDOS_1S = "# E DOS IDOS\n-10.0 0.1 0.3\n-9.0 0.2 0.5\n"
PDOS_1P2 = "# E DOS IDOS\n-10.0 0.05 0.15\n-9.0 0.06 0.21\n"
PDOS_2P10 = "# E DOS IDOS\n-10.0 0.02 0.08\n-9.0 0.03 0.11\n"
PDOS_2D3 = "# E DOS IDOS\n-10.0 0.01 0.02\n-9.0 0.015 0.035\n"
PDOS_1TOTAL = "# E DOS IDOS\n-10.0 0.3 0.9\n-9.0 0.25 1.15\n"


def build_fixtures() -> None:
    """Write the synthetic OpenMX fixtures under tmp_verify/openmx_fixture/."""
    flat = FIXTURE_ROOT / "flat_dos"
    subdir = FIXTURE_ROOT / "subdir_dos"
    bands_dir = FIXTURE_ROOT / "bands"
    atoms_dir = FIXTURE_ROOT / "atoms"
    for d in (flat, subdir / "atom1", subdir / "atom2", bands_dir, atoms_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ---- flat PDOS layout (OpenMX DosMain) ----
    (flat / "sample.DOS.total").write_text(DOS_CONTENT)
    (flat / "sample.PDOS.Tetrahedron.atom1.s1").write_text(PDOS_1S)
    (flat / "sample.PDOS.Tetrahedron.atom1.p2").write_text(PDOS_1P2)
    (flat / "sample.PDOS.Gaussian.atom2.p10").write_text(PDOS_2P10)
    (flat / "sample.PDOS.Tetrahedron.atom2.d3").write_text(PDOS_2D3)
    (flat / "sample.PDOS.Tetrahedron.atom1.total").write_text(PDOS_1TOTAL)

    # ---- legacy atomN/ sub-directory layout ----
    (subdir / "atom1" / "s1").write_text(PDOS_1S)
    (subdir / "atom1" / "p2").write_text(PDOS_1P2)
    (subdir / "atom1" / "d3").write_text(PDOS_2D3)
    (subdir / "atom1" / "total").write_text(PDOS_1TOTAL)
    (subdir / "atom2" / "p10").write_text(PDOS_2P10)

    # ---- .Band fixtures (nspin = 1 and nspin = 2) ----
    spin1 = (
        "2 1 -0.246697\n"
        "1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0\n"
        "1\n"
        "2 0.0 0.0 0.0 0.5 0.0 0.0 G X\n"
        "2 0.0 0.0 0.0\n"
        "-0.5 -0.25\n"
        "2 0.5 0.0 0.0\n"
        "-0.3 -0.2\n"
    )
    (bands_dir / "spin1.Band").write_text(spin1)
    spin2 = (
        "2 2 -0.246697\n"
        "1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0\n"
        "1\n"
        "2 0.0 0.0 0.0 0.5 0.0 0.0 G X\n"
        "2 0.0 0.0 0.0\n"
        "-0.5 -0.25\n"
        "2 0.5 0.0 0.0\n"
        "-0.3 -0.2\n"
        "2 0.0 0.0 0.0\n"
        "-0.6 -0.3\n"
        "2 0.5 0.0 0.0\n"
        "-0.4 -0.25\n"
    )
    (bands_dir / "spin2.Band").write_text(spin2)

    # ---- atomic positions fixtures ----
    ang_dat = (
        "Species.Number 1\n"
        "<Definition.of.Atomic.Species\n"
        "C  C6.0-s3p2d1  C_lda\n"
        "Definition.of.Atomic.Species>\n"
        "Atoms.UnitVectors.Unit Ang\n"
        "<Atoms.UnitVectors\n"
        "3.0 0.0 0.0\n"
        "0.0 3.0 0.0\n"
        "0.0 0.0 6.0\n"
        "Atoms.UnitVectors>\n"
        "Atoms.SpeciesAndCoordinates.Unit Ang\n"
        "<Atoms.SpeciesAndCoordinates\n"
        "1 C 0.5 1.0 1.5 1.0 1.0\n"
        "Atoms.SpeciesAndCoordinates>\n"
    )
    (atoms_dir / "ang_positions.dat").write_text(ang_dat)

    # Unit Ang coordinates but no lattice vectors -> RuntimeError expected.
    ang_no_avecs = ang_dat.replace(
        "Atoms.UnitVectors.Unit Ang\n"
        "<Atoms.UnitVectors\n"
        "3.0 0.0 0.0\n"
        "0.0 3.0 0.0\n"
        "0.0 0.0 6.0\n"
        "Atoms.UnitVectors>\n",
        "",
    )
    (atoms_dir / "ang_no_avecs.dat").write_text(ang_no_avecs)

    # No coordinates anywhere -> original RuntimeError expected.
    no_coords = (
        "Species.Number 1\n"
        "<Definition.of.Atomic.Species\n"
        "C  C6.0-s3p2d1  C_lda\n"
        "Definition.of.Atomic.Species>\n"
    )
    (atoms_dir / "no_coords.dat").write_text(no_coords)


def check_flat_pdos() -> None:
    """(M2) Flat OpenMX PDOS file names are classified per atom/orbital."""
    dos = load_dos_tree(FIXTURE_ROOT / "flat_dos")
    assert dos["total"] is not None, "total DOS not loaded"
    assert list(dos["total"]["E"]) == [-10.0, -9.0]

    assert set(dos["pdos"].keys()) == {1, 2}
    assert list(dos["pdos"][1]["s"].keys()) == ["s1"]
    assert list(dos["pdos"][1]["p"].keys()) == ["p2"]
    assert list(dos["pdos"][2]["p"].keys()) == ["p10"], (
        "p10 key must keep the full number"
    )
    assert list(dos["pdos"][2]["d"].keys()) == ["d3"]
    assert dos["pdos"][1]["total"] is not None, "atom-resolved total PDOS missing"
    print(
        "  [M2-flat] keys: atom1 s={s1} p={p2}; atom2 p={p10} d={d3}; "
        "atom1 total present"
    )


def check_subdir_pdos() -> None:
    """(M2) Legacy atomN/ sub-directory layout still works."""
    dos = load_dos_tree(FIXTURE_ROOT / "subdir_dos")
    assert set(dos["pdos"].keys()) == {1, 2}
    assert list(dos["pdos"][1]["s"].keys()) == ["s1"]
    assert list(dos["pdos"][1]["p"].keys()) == ["p2"]
    assert list(dos["pdos"][1]["d"].keys()) == ["d3"]
    assert list(dos["pdos"][2]["p"].keys()) == ["p10"]
    assert dos["pdos"][1]["total"] is not None
    print("  [M2-subdir] bare names s1/p2/d3/total and p10 classified correctly")


def check_band_spin1() -> None:
    """(M17) nspin=1 keeps the backward-compatible (nk, nband) shape."""
    h2ev = 27.211386245988
    mu_au = -0.246697
    data = parse_dft_band_data(fname_band=str(FIXTURE_ROOT / "bands" / "spin1.Band"))
    assert data["nspin"] == 1
    assert data["bands"].shape == (2, 2), f"expected (nk=2, nband=2), got {data['bands'].shape}"
    assert data["kpts_frac"].shape == (2, 3)
    assert data["kpts_cart"].shape == (2, 3)
    assert np.isclose(data["bands"][0, 0], (-0.5 - mu_au) * h2ev)
    assert np.isclose(data["fermi_energy"], mu_au * h2ev)
    assert len(data["dist"]) == 2
    assert len(data["tick_pos"]) == len(data["tick_label"]) == 2
    print(
        "  [M17-spin1] bands (2, 2), kpts_frac (2, 3), nspin=1 "
        "(backward-compatible)"
    )


def check_band_spin2() -> None:
    """(M17) nspin=2 splits bands by spin; k-points are not duplicated."""
    h2ev = 27.211386245988
    mu_au = -0.246697
    data = parse_dft_band_data(fname_band=str(FIXTURE_ROOT / "bands" / "spin2.Band"))
    assert data["nspin"] == 2
    assert data["bands"].shape == (2, 2, 2), (
        f"expected (nspin=2, nk=2, nband=2), got {data['bands'].shape}"
    )
    # k-points are NOT duplicated per spin
    assert data["kpts_frac"].shape == (2, 3), "kpts_frac must hold one path only"
    assert len(data["dist"]) == 2, "dist must be computed over a single spin"
    # no spurious mid-path jump: dist is monotonically non-decreasing
    assert np.all(np.diff(data["dist"]) >= 0), "dist has a spurious jump"
    # per-spin values: spin0 matches the spin1 fixture, spin1 differs
    assert np.isclose(data["bands"][0, 0, 0], (-0.5 - mu_au) * h2ev)
    assert np.isclose(data["bands"][1, 0, 0], (-0.6 - mu_au) * h2ev)
    assert not np.allclose(data["bands"][0], data["bands"][1])
    print(
        "  [M17-spin2] bands (2, 2, 2), kpts_frac (2, 3) not duplicated, "
        "dist monotonic, spin channels distinct"
    )


def check_ang_positions() -> None:
    """(M18) Unit Ang coordinates are converted via positions_ang @ inv(avecs)."""
    mx = OpenMX()
    result = mx.read_atomic_positions(
        str(FIXTURE_ROOT / "atoms" / "ang_positions.dat")
    )
    avecs = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 6.0]])
    ang = np.array([[0.5, 1.0, 1.5]])
    expected_frac = ang @ np.linalg.inv(avecs)
    assert np.allclose(result["positions_frac"], expected_frac), (
        "frac must equal cart @ inv(avecs)"
    )
    assert np.allclose(result["positions_cart"], ang), (
        "positions_cart should round-trip to the Angstrom input"
    )
    assert result["source"] == "species_coordinates"
    assert mx.n_atoms == 1
    print(
        f"  [M18-ang] frac={result['positions_frac'].tolist()} == cart @ inv(avecs), "
        "no RuntimeError"
    )


def check_ang_without_avecs() -> None:
    """(M18) Ang without lattice vectors raises a clear RuntimeError."""
    mx = OpenMX()
    try:
        mx.read_atomic_positions(str(FIXTURE_ROOT / "atoms" / "ang_no_avecs.dat"))
    except RuntimeError as exc:
        assert "avecs" in str(exc) or "lattice vectors" in str(exc)
        print(f"  [M18-noavecs] RuntimeError raised: {exc}")
        return
    raise AssertionError("expected RuntimeError for Ang coordinates without avecs")


def check_no_coordinates() -> None:
    """(M18) The original RuntimeError is kept when nothing is found."""
    mx = OpenMX()
    try:
        mx.read_atomic_positions(str(FIXTURE_ROOT / "atoms" / "no_coords.dat"))
    except RuntimeError as exc:
        assert "No atomic positions found" in str(exc)
        print(f"  [M18-nocoords] original RuntimeError kept: {exc}")
        return
    raise AssertionError("expected RuntimeError when no coordinates exist")


def main() -> None:
    """Write fixtures and run every check."""
    build_fixtures()
    checks = [
        ("(M2) flat PDOS classification", check_flat_pdos),
        ("(M2) atomN/ sub-directory layout", check_subdir_pdos),
        ("(M17) nspin=1 backward compatibility", check_band_spin1),
        ("(M17) nspin=2 spin splitting", check_band_spin2),
        ("(M18) Unit Ang -> frac via avecs", check_ang_positions),
        ("(M18) Ang without avecs -> RuntimeError", check_ang_without_avecs),
        ("(M18) no coordinates -> RuntimeError", check_no_coordinates),
    ]
    failed = []
    for name, fn in checks:
        try:
            fn()
            print(f"[PASS] {name}")
        except (AssertionError, ValueError) as exc:
            print(f"[FAIL] {name}: {exc}")
            failed.append(name)
    print()
    if failed:
        print(f"RESULT: FAILED ({len(failed)} check(s) failed): {', '.join(failed)}")
        raise SystemExit(1)
    print("RESULT: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
