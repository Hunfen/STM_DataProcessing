import re
import pandas as pd
from pathlib import Path


def load_dos_tree(dos_dir: str | Path = "DOS") -> dict:
    """
    Load total and projected density of states (DOS/PDOS) from a directory structure.

    The expected directory layout is:
    ```
    DOS/
    ├── system.DOS.total          # Total DOS file (optional, must be unique)
    ├── atom1/
    │   ├── s1
    │   ├── p1, p2, ...
    │   ├── d1, d2, ...
    │   └── total                # Total PDOS for atom 1
    ├── atom2/
    │   └── ...
    └── ...
    ```

    Each DOS file should contain three whitespace-separated columns:
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
            - 's': DataFrame (if present)
            - 'p': dict[str, DataFrame] (e.g., 'p1', 'p2')
            - 'd': dict[str, DataFrame] (e.g., 'd1', 'd2')
            - 'total': DataFrame (atom-resolved total PDOS)

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

    dos = {"total": None, "pdos": {}}

    # ---- load system total DOS ----
    total_files = list(dos_dir.glob("*.DOS.*"))
    if len(total_files) == 1:
        dos["total"] = pd.read_csv(
            total_files[0], sep=r"\s+", comment="#", names=["E", "DOS", "IDOS"]
        )
    elif len(total_files) > 1:
        raise RuntimeError(f"Multiple total DOS files found: {total_files}")

    # ---- load PDOS ----
    atom_dirs = sorted(dos_dir.glob("atom*"))
    if not atom_dirs:
        print(f"Warning: No 'atom*' directories found in {dos_dir}")

    for atom_dir in atom_dirs:
        if not atom_dir.is_dir():
            continue
        try:
            atom = int(atom_dir.name.replace("atom", ""))
        except ValueError:
            continue  # skip non-numeric atom directories

        dos["pdos"][atom] = {"p": {}, "d": {}}

        for f in atom_dir.iterdir():
            if not f.is_file():
                continue
            name = f.name
            df = pd.read_csv(f, sep=r"\s+", comment="#", names=["E", "DOS", "IDOS"])

            if re.search(r"\.s\d$", name):
                dos["pdos"][atom]["s"] = df
            elif re.search(r"\.p\d$", name):
                dos["pdos"][atom]["p"][name[-2:]] = df
            elif re.search(r"\.d\d$", name):
                dos["pdos"][atom]["d"][name[-2:]] = df
            else:
                dos["pdos"][atom]["total"] = df

    return dos
