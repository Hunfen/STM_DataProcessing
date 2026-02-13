import numpy as np
import os
import re

# List of valid chemical element symbols for validation
VALID_ELEMENTS = {
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
}


class OpenMX:
    """
    Parser for OpenMX band structure data and output files.

    This class provides methods to read and process band structure data
    from OpenMX output files.

    Attributes
    ----------
    bvecs : np.ndarray or None
        Reciprocal lattice vectors (3x3) in 1/Ångström
    avecs : np.ndarray or None
        Real space lattice vectors (3x3) in Ångström
    folder : str or None
        Folder containing OpenMX files
    systemname : str or None
        System name for OpenMX files
    out_file : str or None
        Path to OpenMX .out file
    dat_file : str or None
        Path to OpenMX .dat file
    atomic_species : dict or None
        Dictionary containing parsed atomic species information
        Format: {element_label: {'element': str, 'orbitals': {'s': int, 'p': int, 'd': int}}}
    n_species : int or None
        Number of valid atomic species (real elements only)
    n_atoms : float or None
        Total number of atoms. Set to np.nan if atomic positions cannot be parsed.
    atomic_positions_data : dict or None
        Cached atomic positions data from the last successful read_atomic_positions call.
    fermi_level : float or None
        Fermi level (chemical potential) in eV, extracted from the last occurrence in the output file
    """

    def __init__(self, folder: str | None = None, systemname: str | None = None):
        """
        Initialize OpenMX parser.

        Parameters
        ----------
        folder : str, optional
            Path to folder containing OpenMX files. If None, files must be loaded manually.
        systemname : str, optional
            System name for OpenMX files. If None, files must be loaded manually.
        """
        self.avecs = None
        self.bvecs = None
        self.folder = folder
        self.systemname = systemname
        self.out_file = None
        self.dat_file = None
        self.atomic_species = None
        self.n_species = None
        self.n_atoms = np.nan  # Initialize as NaN
        self.atomic_positions_data = None  # Cache for atomic positions data
        self.fermi_level = 0.0  # Fermi level in eV

        # Load files if both folder and systemname are provided
        if folder is not None and systemname is not None:
            self.load_from_systemname(folder, systemname)

    @staticmethod
    def _parse_fermi_level(lines: list[str]) -> float | None:
        """
        Parse Fermi level (Chemical Potential) from OpenMX file lines.

        Parameters
        ----------
        lines : list of str
            Lines from the OpenMX file.

        Returns
        -------
        float or None
            Fermi level in eV (converted from Hartree), or None if not found.

        Notes
        -----
        The function finds the last occurrence of "Chemical Potential (Hartree)"
        and converts it to eV using the conversion factor 1 Hartree = 27.211386245988 eV.
        """
        hartree_to_ev = 27.211386245988
        chemical_potential = None

        for line in lines:
            if "Chemical Potential (Hartree)" in line:
                try:
                    # Extract the value after the equals sign
                    value_str = line.split("=")[1].strip()
                    chemical_potential = float(value_str)
                except (IndexError, ValueError) as e:
                    print(f"Warning: Could not parse chemical potential: {e}")
                    continue

        if chemical_potential is not None:
            return chemical_potential * hartree_to_ev
        return None

    @staticmethod
    def _parse_unit_cell_vectors(lines: list[str]) -> np.ndarray | None:
        """
        Parse unit cell vectors from OpenMX file lines.

        Parameters
        ----------
        lines : list of str
            Lines from the OpenMX file.

        Returns
        -------
        np.ndarray or None
            (3,3) array of real space lattice vectors, or None if not found.
        """
        avecs_start_idx = -1
        avecs_end_idx = -1

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if "<Atoms.UnitVectors" in line_stripped:
                avecs_start_idx = i + 1
            elif "Atoms.UnitVectors>" in line_stripped:
                avecs_end_idx = i

        if avecs_start_idx != -1 and avecs_end_idx != -1:
            avecs_lines = []
            for i in range(avecs_start_idx, avecs_end_idx):
                line_stripped = lines[i].strip()
                if line_stripped and not line_stripped.startswith("#"):
                    avecs_lines.append(line_stripped)

            if len(avecs_lines) >= 3:
                try:
                    a1 = np.array(list(map(float, avecs_lines[0].split())))
                    a2 = np.array(list(map(float, avecs_lines[1].split())))
                    a3 = np.array(list(map(float, avecs_lines[2].split())))
                    return np.array([a1, a2, a3])
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse unit cell vectors: {e}")

        return None

    @staticmethod
    def _parse_reciprocal_vectors(lines: list[str]) -> np.ndarray | None:
        """
        Parse reciprocal lattice vectors from OpenMX file lines.

        Parameters
        ----------
        lines : list of str
            Lines from the OpenMX file.

        Returns
        -------
        np.ndarray or None
            (3,3) array of reciprocal lattice vectors, or None if not found.
        """
        b1 = b2 = b3 = None

        for line in lines:
            s = line.strip()
            if "Reciprocal vector b1" in s:
                b1 = OpenMX._parse_vector_line(s)
            elif "Reciprocal vector b2" in s:
                b2 = OpenMX._parse_vector_line(s)
            elif "Reciprocal vector b3" in s:
                b3 = OpenMX._parse_vector_line(s)

        if b1 is not None and b2 is not None and b3 is not None:
            return np.vstack([b1, b2, b3])

        return None

    @staticmethod
    def _parse_atomic_species(lines: list[str]) -> dict:
        """
        Parse atomic species definition from OpenMX file lines.

        Parameters
        ----------
        lines : list of str
            Lines from the OpenMX file.

        Returns
        -------
        dict
            Dictionary containing parsed atomic species information.
        """
        # Find the atomic species block
        start_idx = -1
        end_idx = -1
        species_number = None

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if line_stripped.startswith("Species.Number"):
                # Parse Species.Number
                try:
                    species_number = int(line_stripped.split()[1])
                except (IndexError, ValueError) as err:
                    raise RuntimeError(
                        f"Invalid Species.Number line: {line_stripped}"
                    ) from err

            if "<Definition.of.Atomic.Species" in line_stripped:
                start_idx = i + 1
            elif "Definition.of.Atomic.Species>" in line_stripped:
                end_idx = i

        if start_idx == -1 or end_idx == -1:
            raise RuntimeError("Atomic species block not found")

        if species_number is None:
            raise RuntimeError("Species.Number not found")

        # Extract the species definition lines
        species_lines = []
        for i in range(start_idx, end_idx):
            line_stripped = lines[i].strip()
            if line_stripped and not line_stripped.startswith("#"):
                species_lines.append(line_stripped)

        # Validate that we have the expected number of lines
        if len(species_lines) != species_number:
            print(
                f"Warning: Species.Number={species_number} but found {len(species_lines)} species lines"
            )

        # Parse each species line
        valid_species = []
        raw_lines = []

        for line in species_lines:
            raw_lines.append(line)
            parts = line.split()
            if len(parts) < 3:
                print(f"Warning: Skipping invalid species line: {line}")
                continue

            label = parts[0]
            basis_info = parts[1]
            pseudopotential = parts[2]

            # Check if the label is a valid element
            if label not in VALID_ELEMENTS:
                print(f"Skipping non-element species: {label}")
                continue

            # Parse orbital information from basis_info
            # Format: <element><number><H/S optional>-s<i>p<j>d<k>
            orbitals = OpenMX._parse_orbital_basis_static(basis_info, label)

            if orbitals is None:
                print(
                    f"Warning: Could not parse orbital basis for {label}: {basis_info}"
                )
                continue

            species_dict = {
                "label": label,
                "element": label,
                "orbitals": orbitals,
                "pseudopotential": pseudopotential,
                "basis_info": basis_info,
            }
            valid_species.append(species_dict)

        n_valid_species = len(valid_species)
        print(
            f"Found {n_valid_species} valid atomic species out of {len(species_lines)} total lines"
        )

        result = {
            "species_list": valid_species,
            "n_species": n_valid_species,
            "raw_lines": raw_lines,
        }
        return result

    @staticmethod
    def _parse_vector_line(line: str) -> np.ndarray:
        """Helper to extract a vector from a line like:
        '#  Reciprocal vector b1 (1/Ang):   2.55414  1.47463  0.00000'
        """
        right = line.split(":", 1)[1]
        vals = right.split()
        return np.array(list(map(float, vals[:3])))

    @staticmethod
    def _parse_orbital_basis_static(basis_info: str, element: str) -> dict | None:
        """
        Parse orbital basis string to extract s, p, d orbital counts.

        Parameters
        ----------
        basis_info : str
            Basis string like "C6.0-s3p2d2" or "Li8.0-s3p2d1"
        element : str
            Element symbol for validation

        Returns
        -------
        dict or None
            Dictionary with 's', 'p', 'd' keys and integer values, or None if parsing fails
        """
        try:
            # Split on '-' to separate element part from orbital part
            if "-" not in basis_info:
                return None

            orbital_part = basis_info.split("-", 1)[1]

            # Initialize orbital counts
            orbitals = {"s": 0, "p": 0, "d": 0}

            # Parse s, p, d orbitals using regex
            # Look for patterns like s3, p2, d2
            s_match = re.search(r"s(\d+)", orbital_part)
            p_match = re.search(r"p(\d+)", orbital_part)
            d_match = re.search(r"d(\d+)", orbital_part)

            if s_match:
                orbitals["s"] = int(s_match.group(1))
            if p_match:
                orbitals["p"] = int(p_match.group(1))
            if d_match:
                orbitals["d"] = int(d_match.group(1))

            return orbitals

        except Exception as e:
            print(
                f"Error parsing orbital basis '{basis_info}' for element {element}: {e}"
            )
            return None

    def _create_positions_dict(self, positions_frac, elements, spin_weights, source):
        """Helper method to create the standardized positions dictionary."""
        n_atoms = len(positions_frac)

        # Handle spin weights - if not provided, use np.nan
        if spin_weights is None:
            spin_weights = np.full((n_atoms, 2), np.nan, dtype=np.float64)

        # Calculate Cartesian coordinates if lattice vectors are available
        positions_cart = None
        if self.avecs is not None:
            # Ensure both arrays are float64 for maximum precision
            positions_frac_array = np.array(positions_frac, dtype=np.float64)
            avecs_array = np.array(self.avecs, dtype=np.float64)
            positions_cart = np.dot(positions_frac_array, avecs_array)

        return {
            "positions_frac": np.array(positions_frac, dtype=np.float64),
            "positions_cart": positions_cart,
            "elements": elements,
            "spin_weights": np.array(spin_weights, dtype=np.float64),
            "source": source,
        }

    @staticmethod
    def _parse_final_structure_positions(lines: list[str]) -> dict:
        """
        Parse final structure fractional coordinates from OpenMX output.

        Parameters
        ----------
        lines : list of str
            Lines from the OpenMX file.

        Returns
        -------
        dict
            Dictionary with 'positions_frac' and 'elements' keys, or None values if not found.
        """
        # Find the final structure section
        start_idx = -1
        for i, line in enumerate(lines):
            if "Fractional coordinates of the final structure" in line:
                # Look for the actual data lines after the header
                start_idx = i + 3  # Skip the header lines
                break

        if start_idx == -1:
            return {"positions_frac": None, "elements": None}

        positions = []
        elements = []

        # Parse atom lines until we hit an empty line or non-numeric line
        for i in range(start_idx, len(lines)):
            line = lines[i].strip()
            if not line or line.startswith("*") or not line[0].isdigit():
                break

            parts = line.split()
            if len(parts) < 5:  # Need at least: index, element, x, y, z
                continue

            try:
                element = parts[1]
                # Use float() which gives double precision (64-bit)
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                positions.append([x, y, z])
                elements.append(element)
            except (ValueError, IndexError):
                continue

        if not positions:
            return {"positions_frac": None, "elements": None}

        return {"positions_frac": positions, "elements": elements}

    @staticmethod
    def _parse_species_and_coordinates(lines: list[str]) -> dict | None:
        """
        Parse Atoms.SpeciesAndCoordinates block from OpenMX file.

        Parameters
        ----------
        lines : list of str
            Lines from the OpenMX file.

        Returns
        -------
        dict or None
            Dictionary with 'positions_frac', 'elements', and 'spin_weights' keys,
            or None if the block is not found.
        """
        # Find the coordinate unit
        coord_unit = None
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.startswith("Atoms.SpeciesAndCoordinates.Unit"):
                parts = line_stripped.split()
                if len(parts) >= 2:
                    coord_unit = parts[1].upper()
                break

        if coord_unit not in ["FRAC", "ANG"]:
            # If unit is not specified or invalid, assume FRAC as default
            coord_unit = "FRAC"

        # Find the species and coordinates block
        start_idx = -1
        end_idx = -1
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if "<Atoms.SpeciesAndCoordinates" in line_stripped:
                start_idx = i + 1
            elif "Atoms.SpeciesAndCoordinates>" in line_stripped:
                end_idx = i

        if start_idx == -1 or end_idx == -1:
            return None

        positions = []
        elements = []
        spin_weights = []

        # Parse atom lines
        for i in range(start_idx, end_idx):
            line_stripped = lines[i].strip()
            if not line_stripped or line_stripped.startswith("#"):
                continue

            parts = line_stripped.split()
            if len(parts) < 5:  # Need at least: index, element, x, y, z
                continue

            try:
                element = parts[1]
                # Use float() for double precision
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])

                # Handle spin weights if available (columns 6 and 7)
                up_spin = down_spin = np.nan
                if len(parts) >= 7:
                    try:
                        up_spin = float(parts[5])
                        down_spin = float(parts[6])
                    except (ValueError, IndexError):
                        pass

                positions.append([x, y, z])
                elements.append(element)
                spin_weights.append([up_spin, down_spin])

            except (ValueError, IndexError):
                continue

        if not positions:
            return None

        # If coordinates are in Angstrom, convert to fractional
        if coord_unit == "ANG":
            # Note: We cannot convert Angstrom to fractional without lattice vectors
            # So we'll keep them as-is and let the caller handle the conversion
            # This is a limitation - the user would need to provide lattice vectors separately
            pass

        return {
            "positions_frac": positions if coord_unit == "FRAC" else None,
            "positions_ang": positions if coord_unit == "ANG" else None,
            "elements": elements,
            "spin_weights": spin_weights,
        }

    def read_atomic_species_from_out(self, fname: str | None = None) -> dict:
        """
        Read atomic species definition from OpenMX .out/.dat file.
        This method calls read_openmx_file to ensure all properties are parsed.

        Parameters
        ----------
        fname : str, optional
            Path to the OpenMX output/dat file. If None and out_file/dat_file are set
            during initialization, uses the available file path.

        Returns
        -------
        atomic_species : dict
            Dictionary containing parsed atomic species information with keys:
            - 'species_list': list of valid species dictionaries
            - 'n_species': number of valid atomic species
            - 'raw_lines': list of raw species definition lines

            Each species dictionary contains:
            - 'label': original label from first column
            - 'element': validated element symbol
            - 'orbitals': dict with 's', 'p', 'd' orbital counts

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        RuntimeError
            If atomic species block is not found or malformed.
        ValueError
            If no valid file path is available.
        """
        # Use read_openmx_file to get all data including atomic species
        result = self.read_openmx_file(fname)
        return {
            "species_list": result.get("species_list", []),
            "n_species": result.get("n_species", 0),
            "raw_lines": result.get("raw_lines", []),
        }

    def read_bvecs_from_out(self, fname: str | None = None) -> np.ndarray:
        """
        Read reciprocal lattice vectors from an OpenMX .out/.log/.dat file.
        This method calls read_openmx_file to ensure all properties are parsed.

        Parameters
        ----------
        fname : str, optional
            Path to the OpenMX output/dat files. If None and out_file/dat_file are set
            during initialization, uses the available file path.

        Returns
        -------
        bvecs : (3,3) ndarray
            Reciprocal lattice vectors b1, b2, b3 in 1/Ångström.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        RuntimeError
            If reciprocal vectors are not found in the file.
        ValueError
            If no valid file path is available.
        """
        # Use read_openmx_file to get all data including bvecs
        result = self.read_openmx_file(fname)
        if result.get("bvecs") is None:
            raise RuntimeError("Reciprocal vectors not found in  the file")

        self.bvecs = result["bvecs"]
        return self.bvecs

    def read_atomic_positions(self, fname: str | None = None) -> dict:
        """
        Read atomic positions from OpenMX .out/.dat file with priority order.

        Priority order for atomic positions:
        1. Final structure fractional coordinates from .out file (highest priority)
        2. Atoms.SpeciesAndCoordinates block from .out file
        3. Atoms.SpeciesAndCoordinates block from .dat file (lowest priority)

        Spin weights are only available from sources 2 and 3.

        Parameters
        ----------
        fname : str, optional
            Path to the OpenMX output/dat file. If None and out_file/dat_file are set
            during initialization, uses the available file path.

        Returns
        -------
        dict
            Dictionary containing parsed atomic positions information:
            - 'positions_frac': (n_atoms, 3) array of fractional coordinates
            - 'positions_cart': (n_atoms, 3) array of Cartesian coordinates (if avecs available)
            - 'elements': list of element symbols
            - 'spin_weights': (n_atoms, 2) array of [up_spin, down_spin] weights (np.nan if not available)
            - 'source': string indicating which source was used ('final_structure', 'species_coordinates', or 'dat_file')

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        RuntimeError
            If no atomic positions can be found in any of the sources.
        ValueError
            If no valid file path is available.
        """
        # First, ensure we have file paths and read basic information
        result = self.read_openmx_file(fname)

        # Try to get positions from final structure (priority 1)
        final_positions_result = self._parse_final_structure_positions(
            result.get("lines", [])
        )
        if final_positions_result["positions_frac"] is not None:
            # We have final structure positions, now try to get spin weights from species coordinates
            species_result = self._parse_species_and_coordinates(
                result.get("lines", [])
            )
            spin_weights = species_result["spin_weights"] if species_result else None

            positions_dict = self._create_positions_dict(
                final_positions_result["positions_frac"],
                final_positions_result["elements"],
                spin_weights,
                "final_structure",
            )
            # Set the atomic positions data and atom count
            self.atomic_positions_data = positions_dict
            self.n_atoms = len(positions_dict["elements"])
            return positions_dict

        # Try to get positions from species and coordinates block (priority 2)
        species_result = self._parse_species_and_coordinates(result.get("lines", []))
        if species_result and species_result["positions_frac"] is not None:
            positions_dict = self._create_positions_dict(
                species_result["positions_frac"],
                species_result["elements"],
                species_result["spin_weights"],
                "species_coordinates",
            )
            # Set the atomic positions data and atom count
            self.atomic_positions_data = positions_dict
            self.n_atoms = len(positions_dict["elements"])
            return positions_dict

        # If we have a .dat file and haven't tried it yet, try reading from .dat
        if (
            fname is None
            and self.dat_file is not None
            and os.path.exists(self.dat_file)
        ):
            try:
                with open(self.dat_file) as f:
                    dat_lines = f.readlines()
                dat_species_result = self._parse_species_and_coordinates(dat_lines)
                if (
                    dat_species_result
                    and dat_species_result["positions_frac"] is not None
                ):
                    positions_dict = self._create_positions_dict(
                        dat_species_result["positions_frac"],
                        dat_species_result["elements"],
                        dat_species_result["spin_weights"],
                        "dat_file",
                    )
                    # Set the atomic positions data and atom count
                    self.atomic_positions_data = positions_dict
                    self.n_atoms = len(positions_dict["elements"])
                    return positions_dict
            except FileNotFoundError:
                pass

        # If we reach here, no atomic positions were found
        self.n_atoms = np.nan
        self.atomic_positions_data = None
        raise RuntimeError("No atomic positions found in any of the available sources")

    def load_from_systemname(self, folder: str, systemname: str) -> None:
        """
        Load OpenMX files from folder and system name.

        Parameters
        ----------
        folder : str
            Path to folder containing OpenMX files.
        systemname : str
            System name for OpenMX files.

        Notes
        -----
        This method only sets file paths and prints warnings if files don't exist.
        It does not raise exceptions for missing files since they may not be needed
        for all operations.
        """
        # Construct file paths
        self.out_file = os.path.join(folder, f"{systemname}.out")
        self.dat_file = os.path.join(folder, f"{systemname}.dat")

        # Check if files exist
        if not os.path.exists(self.out_file):
            print(f"Warning: .out file not found: {self.out_file}")
        else:
            print(f"  Found .out file: {self.out_file}")

        if not os.path.exists(self.dat_file):
            print(f"Warning: .dat file not found: {self.dat_file}")
        else:
            print(f"  Found .dat file: {self.dat_file}")

        self.folder = folder
        self.systemname = systemname

        print(f"  OpenMX System.Name: {systemname}")
        print(f"  Folder: {folder}")

    def read_openmx_file(self, fname: str | None = None) -> dict:
        """
        Read OpenMX .out/.dat file and extract various properties.

        Parameters
        ----------
        fname : str, optional
            Path to the OpenMX output/dat file. If None and out_file/dat_file are set
            during initialization, uses the available file path.

        Returns
        -------
        dict
            Dictionary containing parsed information:
            - 'avecs': (3,3) array of real space lattice vectors (if found)
            - 'bvecs': (3,3) array of reciprocal lattice vectors (if found)
            - 'atomic_species': dict with species information
            - 'n_species': number of valid atomic species
            - 'fermi_level': fermi level in eV (if found)
            - 'lines': list of all lines from the file (for internal parsing)
        """
        # Determine the file path to use
        if fname is None:
            if self.out_file is not None and os.path.exists(self.out_file):
                fname = self.out_file
                print(f"Using auto-detected .out file: {fname}")
            elif self.dat_file is not None and os.path.exists(self.dat_file):
                fname = self.dat_file
                print(f"Using auto-detected .dat file: {fname}")
            elif self.folder is not None and self.systemname is not None:
                # Try to construct paths even if they weren't set before
                out_path = os.path.join(self.folder, f"{self.systemname}.out")
                dat_path = os.path.join(self.folder, f"{self.systemname}.dat")
                if os.path.exists(out_path):
                    fname = out_path
                    print(f"Using constructed .out file: {fname}")
                elif os.path.exists(dat_path):
                    fname = dat_path
                    print(f"Using constructed .dat file: {fname}")
                else:
                    raise ValueError(
                        "No valid file path provided and no .out/.dat files found. "
                        "Please provide fname or ensure OpenMX files exist."
                    )
            else:
                raise ValueError(
                    "fname not provided and no file paths set during initialization. "
                    "Please provide fname or initialize with folder and systemname."
                )
        else:
            print(f"Using provided file: {fname}")

        # Read the file
        try:
            with open(fname) as f:
                lines = f.readlines()
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {fname}") from e

        # Parse different sections
        result = {"lines": lines}

        # Parse unit cell vectors
        result["avecs"] = self._parse_unit_cell_vectors(lines)
        if result["avecs"] is not None:
            self.avecs = result["avecs"]

        # Parse reciprocal vectors
        result["bvecs"] = self._parse_reciprocal_vectors(lines)
        if result["bvecs"] is not None:
            self.bvecs = result["bvecs"]

        # Parse atomic species
        species_result = self._parse_atomic_species(lines)
        result.update(species_result)

        # Parse fermi level
        result["fermi_level"] = self._parse_fermi_level(lines)
        if result["fermi_level"] is not None:
            self.fermi_level = result["fermi_level"]

        # Update instance variables
        self.atomic_species = species_result.get("species_list")
        self.n_species = species_result.get("n_species")

        return result
