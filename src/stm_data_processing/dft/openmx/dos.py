import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# OpenMX PDOS orbital labels: angular-momentum letter plus a 1- or 2-digit
# component index, e.g. 's1', 'p2', 'p10', 'd3'.  The full label (including
# the multi-digit number) is the dictionary key, so 'p10' is never truncated
# to 'p1' + '0'.
_ORBITAL_RE = re.compile(r"^([spdfg])(\d+)$")

# Flat OpenMX DosMain PDOS file names, e.g.
#   <System>.PDOS.Tetrahedron.atom1.s1
#   <System>.PDOS.Gaussian.atom2.p10
_PDOS_FLAT_RE = re.compile(r"\.PDOS\..*\.atom(\d+)\.([A-Za-z]+\d*)$")


def _classify_orbital(label: str) -> tuple[str, str] | None:
    """Classify a bare orbital label into (kind, key).

    Recognized labels are 'total' and orbital labels of the form
    '<letter><number>' (s/p/d/f/g), e.g. 's1', 'p2', 'p10', 'd3'.  The key is
    the full label so multi-digit orbital numbers are preserved.  Returns None
    for unrecognized labels.
    """
    if label == "total":
        return "total", "total"
    match = _ORBITAL_RE.match(label)
    if match is not None:
        return match.group(1), label
    return None


def load_dos_tree(dos_dir: str | Path = "DOS") -> dict:
    """
    Load total and projected density of states (DOS/PDOS) from a directory.

    Two PDOS layouts are supported:

    1. Flat OpenMX DosMain output (Tetrahedron or Gaussian broadening):
       DOS/
       |-- system.DOS.total          # Total DOS file (optional, must be unique)
       |-- system.PDOS.Tetrahedron.atom1.s1
       |-- system.PDOS.Tetrahedron.atom1.p2
       |-- system.PDOS.Gaussian.atom2.p10
       `-- system.PDOS.Tetrahedron.atom1.total   # Atom-resolved total PDOS
    2. Legacy atomN/ sub-directories:
       DOS/
       |-- atom1/
       |   |-- s1
       |   |-- p2
       |   |-- d3
       |   `-- total                # Total PDOS for atom 1
       `-- atom2/
           `-- p10

    The total DOS file is recognized as <System>.DOS or <System>.DOS.*
    (excluding PDOS files).  Each DOS file contains three whitespace-separated
    columns:
    - Energy (eV)
    - DOS (states/eV)
    - Integrated DOS (IDOS)

    Lines starting with '#' are treated as comments.

    Parameters
    ----------
    dos_dir : str or Path, optional
        Path to the DOS directory. Default is "DOS".

    Returns
    -------
    dict
        A dictionary with keys:
        - 'total': pd.DataFrame with columns ['E', 'DOS', 'IDOS'] or None
        - 'pdos': dict[int, dict] where each atom has:
            - 's': dict[str, DataFrame] keyed by full orbital label (e.g. 's1')
            - 'p': dict[str, DataFrame] (e.g. 'p1', 'p2', 'p10')
            - 'd': dict[str, DataFrame] (e.g. 'd1', 'd3')
            - 'total': DataFrame (atom-resolved total PDOS) or None

    Raises
    ------
    RuntimeError
        If multiple total DOS files are found.
    FileNotFoundError
        If the DOS directory does not exist.
    """
    dos_dir = Path(dos_dir)

    if not dos_dir.exists():
        raise FileNotFoundError(f"DOS directory not found: {dos_dir}")

    dos: dict = {"total": None, "pdos": {}}

    def _load_df(path: Path) -> pd.DataFrame:
        return pd.read_csv(
            path, sep=r"\s+", comment="#", names=["E", "DOS", "IDOS"]
        )

    def _store_pdos(atom: int, kind: str, key: str, df: pd.DataFrame) -> None:
        entry = dos["pdos"].setdefault(
            atom, {"s": {}, "p": {}, "d": {}, "total": None}
        )
        if kind == "total":
            entry["total"] = df
        else:
            entry.setdefault(kind, {})[key] = df

    # ---- load system total DOS ----
    total_files = sorted(
        p
        for p in dos_dir.iterdir()
        if p.is_file()
        and re.search(r"\.DOS($|\.)", p.name) is not None
        and ".PDOS." not in p.name
    )
    if len(total_files) == 1:
        dos["total"] = _load_df(total_files[0])
    elif len(total_files) > 1:
        raise RuntimeError(f"Multiple total DOS files found: {total_files}")

    # ---- load PDOS: flat OpenMX files in the directory root ----
    for f in sorted(p for p in dos_dir.iterdir() if p.is_file()):
        flat = _PDOS_FLAT_RE.search(f.name)
        if flat is None:
            continue
        atom = int(flat.group(1))
        classified = _classify_orbital(flat.group(2))
        if classified is None:
            logger.warning("Unrecognized PDOS orbital label: %s", f.name)
            continue
        kind, key = classified
        _store_pdos(atom, kind, key, _load_df(f))

    # ---- load PDOS: legacy atomN/ sub-directories ----
    atom_dirs = sorted(dos_dir.glob("atom*"))
    if not atom_dirs:
        logger.debug("No 'atom*' directories found in %s", dos_dir)

    for atom_dir in atom_dirs:
        if not atom_dir.is_dir():
            continue
        try:
            atom = int(atom_dir.name.replace("atom", ""))
        except ValueError:
            continue  # skip non-numeric atom directories

        for f in sorted(p for p in atom_dir.iterdir() if p.is_file()):
            classified = _classify_orbital(f.name)
            if classified is None:
                logger.warning(
                    "Unrecognized PDOS file in %s: %s", atom_dir, f.name
                )
                continue
            kind, key = classified
            _store_pdos(atom, kind, key, _load_df(f))

    return dos
