"""Wannier90 HR data loading utilities."""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from stm_data_processing.io.lattice_loader import LatticeLoader


class Wannier90HRLoader:
    """
    Loader for Wannier90 HR Hamiltonian data files.

    This class handles all file I/O operations for Wannier90 data,
    including parsing HR files and extracting reciprocal lattice vectors.

    Public API
    ----------
    load(folder, seedname) : Load all HR data from Wannier90 files
    """

    logger = logging.getLogger(__name__)

    def __init__(self) -> None:
        """Initialize the data loader."""
        pass

    @staticmethod
    def load(folder: str, seedname: str) -> dict[str, Any]:
        """
        Load Wannier90 HR Hamiltonian data from file.

        Parameters
        ----------
        folder : str
            Directory containing the HR file.
        seedname : str
            Base name of the Wannier90 files (without extension).

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - num_wann : int
            - r_list : np.ndarray (nrpts, 3) int32
            - h_list : np.ndarray (nrpts, nw, nw) complex128
            - ndegen : np.ndarray (nrpts,) float64
            - h_list_flat : np.ndarray (nrpts, nw*nw) complex128
            - bvecs : np.ndarray | None (3, 3) reciprocal lattice vectors

        Raises
        ------
        FileNotFoundError
            If the HR file does not exist.
        RuntimeError
            If the file format is invalid.
        """
        folder_p = Path(folder)
        hr_file = folder_p / f"{seedname}_hr.dat"
        if not hr_file.exists():
            raise FileNotFoundError(f"HR file not found: {hr_file}")

        Wannier90HRLoader.logger.info(
            "[Wannier90HRLoader] Loading HR data from: %s", hr_file
        )

        bvecs = Wannier90HRLoader._load_bvecs(folder_p, seedname)
        num_wann, r_list, h_list, ndegen = Wannier90HRLoader._parse_hr_file(hr_file)

        h_list_flat = h_list.reshape(len(r_list), num_wann * num_wann)

        Wannier90HRLoader.logger.info(
            "[Wannier90HRLoader] Loaded HR with num_wann=%d, nrpts=%d",
            num_wann,
            len(r_list),
        )

        return {
            "num_wann": num_wann,
            "r_list": r_list,
            "h_list": h_list,
            "ndegen": ndegen,
            "h_list_flat": h_list_flat,
            "bvecs": bvecs,
        }

    @staticmethod
    def _load_bvecs(folder: Path, seedname: str) -> np.ndarray | None:
        """
        Attempt to load reciprocal lattice vectors from .wout or .out file.

        Parameters
        ----------
        folder : Path
            Directory containing the output files.
        seedname : str
            Base name of the Wannier90 files.

        Returns
        -------
        np.ndarray | None
            Reciprocal lattice vectors (3, 3) if found, else None.
        """
        wout_file = folder / f"{seedname}.wout"
        out_file = folder / f"{seedname}.out"

        lattice_obj = None
        if wout_file.exists():
            lattice_obj = LatticeLoader.create_lattice(filename=wout_file)
        elif out_file.exists():
            lattice_obj = LatticeLoader.create_lattice(filename=out_file)

        return None if lattice_obj is None else lattice_obj.bvecs

    @staticmethod
    def _parse_hr_file(
        filename: Path,
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """
        Parse Wannier90 seedname_hr.dat file.

        Parameters
        ----------
        filename : Path
            Path to the HR file.

        Returns
        -------
        tuple[int, np.ndarray, np.ndarray, np.ndarray]
            (num_wann, r_list, h_list, ndegen)

        Raises
        ------
        RuntimeError
            If the file format is invalid.
        """
        with filename.open("r") as f:
            first_line = f.readline()
            first_stripped = first_line.strip().lower()

            if first_stripped.startswith("written on"):
                num_wann = int(f.readline().split()[0])
                nrpts = int(f.readline().split()[0])
            else:
                num_wann = int(first_line.split()[0])
                nrpts = int(f.readline().split()[0])

            ndegen: list[int] = []
            while len(ndegen) < nrpts:
                line = f.readline()
                if not line:
                    raise RuntimeError("Unexpected EOF while reading ndegen list.")
                ndegen.extend([int(x) for x in line.split()])
            ndegen_arr = np.array(ndegen[:nrpts], dtype=np.float64)

            r_list: list[tuple[int, int, int]] = []
            h_list: list[np.ndarray] = []

            current_r: tuple[int, int, int] | None = None
            current_h: np.ndarray | None = None

            nlines = nrpts * num_wann * num_wann
            for _ in range(nlines):
                parts = f.readline().split()
                if len(parts) < 7:
                    raise RuntimeError("Unexpected EOF or malformed hr.dat line.")

                r1, r2, r3 = map(int, parts[:3])
                m, n = map(int, parts[3:5])
                re, im = map(float, parts[5:7])

                r = (r1, r2, r3)
                if current_r is None or r != current_r:
                    if current_r is not None and current_h is not None:
                        r_list.append(current_r)
                        h_list.append(current_h)

                    current_r = r
                    current_h = np.zeros((num_wann, num_wann), dtype=np.complex128)

                i = m - 1
                j = n - 1
                current_h[i, j] = re + 1j * im

            if current_r is not None and current_h is not None:
                r_list.append(current_r)
                h_list.append(current_h)

        if len(r_list) != nrpts:
            raise RuntimeError(
                f"nrpts mismatch: read {len(r_list)} r-points, expected {nrpts}"
            )

        r_array = np.array(r_list, dtype=np.int32)
        h_array = np.stack(h_list, axis=0).astype(np.complex128)

        return num_wann, r_array, h_array, ndegen_arr
