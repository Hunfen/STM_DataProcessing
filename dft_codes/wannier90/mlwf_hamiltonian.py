import os
import re

import numpy as np

# Try to import cupy for GPU support
try:
    import cupy as cp

    CUPY_AVAILABLE = True
    print("CuPy is available. GPU acceleration enabled.")
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not found. Using CPU-only mode.")


class MLWFHamiltonian:
    """
    Class for handling Wannier90 Hamiltonian files.

    Attributes
    ----------
    num_wann : int
        Number of Wannier functions
    r_list : np.ndarray
        Integer lattice vectors R (nrpts, 3)
    h_list : np.ndarray
        Complex Hamiltonian matrices H(R) (nrpts, num_wann, num_wann)
    ndegen : np.ndarray
        Integer degeneracy weights (nrpts,)
    use_gpu : bool
        Whether to use GPU acceleration (if available)
    folder : str or None
        Folder containing Wannier90 files
    seedname : str or None
        Seedname for Wannier90 files
    """

    def __init__(
        self,
        folder: str | None = None,
        seedname: str | None = None,
        use_gpu: bool = True,
    ):
        """
        Initialize MLWFHamiltonian.

        Parameters
        ----------
        folder : str, optional
            Path to folder containing Wannier90 files. If None, files must be loaded manually.
        seedname : str, optional
            Seedname for Wannier90 files. If None, files must be loaded manually.
        use_gpu : bool, optional
            Whether to use GPU acceleration if available. Default is True.
        """
        self.num_wann = None
        self.r_list = None
        self.h_list = None
        self.ndegen = None
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.folder = folder
        self.seedname = seedname

        # GPU arrays (initialized only when needed)
        self.r_list_gpu = None
        self.h_list_gpu = None
        self.ndegen_gpu = None

        # Load files if both folder and seedname are provided
        if folder is not None and seedname is not None:
            self.load_from_seedname(folder, seedname)

    def load_from_seedname(self, folder: str, seedname: str) -> None:
        """
        Load Wannier90 Hamiltonian files from folder and seedname.

        Parameters
        ----------
        folder : str
            Path to folder containing Wannier90 files.
        seedname : str
            Seedname for Wannier90 files.

        Raises
        ------
        FileNotFoundError
            If the required files do not exist.
        """
        # Construct file paths
        hr_file = os.path.join(folder, f"{seedname}_hr.dat")

        # Check if hr.dat file exists
        if not os.path.exists(hr_file):
            raise FileNotFoundError(f"hr.dat file not found: {hr_file}")

        # Try to find wlog_file (wout or out file)
        wout_file = os.path.join(folder, f"{seedname}.wout")
        out_file = os.path.join(folder, f"{seedname}.out")

        wlog_file = None
        if os.path.exists(wout_file):
            wlog_file = wout_file
            print(f"  Found wout file: {wout_file}")
        elif os.path.exists(out_file):
            wlog_file = out_file
            print(f"  Found out file: {out_file}")
        else:
            print(
                f"Warning: Neither {seedname}.wout nor {seedname}.out found in {folder}"
            )
            print(
                "  You may need to provide wlog_file parameter separately for calculate_contourmap()"
            )

        self.folder = folder
        self.seedname = seedname
        self.wlog_file = wlog_file

        # Load hr.dat file
        self.load_hr_file(hr_file)

        print(f"  Wannier90 seedname: {seedname}")
        print(f"  Folder: {folder}")
        print(f"  hr.dat file: {hr_file}")
        if wlog_file:
            print(f"  wlog_file (auto-detected): {wlog_file}")

    def load_hr_file(self, filename: str) -> None:
        """
        Load Wannier90 Hamiltonian from hr.dat file.

        Parameters
        ----------
        filename : str
            Path to hr.dat file.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        RuntimeError
            If file format is invalid or inconsistent.
        """
        try:
            with open(filename) as f:
                # Check first line for timestamp header (new format)
                first_line = f.readline()
                first_stripped = first_line.strip().lower()
                if first_stripped.startswith("written on"):
                    # Next line contains num_wann
                    num_wann = int(f.readline().split()[0])
                    nrpts = int(f.readline().split()[0])
                else:
                    # Old format: first line contains num_wann
                    num_wann = int(first_line.split()[0])
                    nrpts = int(f.readline().split()[0])

                # Read ndegen block (up to 15 values per line)
                ndegen = []
                while len(ndegen) < nrpts:
                    line = f.readline()
                    if not line:
                        raise RuntimeError("Unexpected EOF while reading ndegen block.")
                    ndegen.extend([int(x) for x in line.split()])
                ndegen = np.array(ndegen[:nrpts], dtype=int)

                # Read nrpts * num_wann^2 lines of H(R) data
                r_list = []
                h_list = []  # Each R corresponds to a num_wann x num_wann matrix
                current_r = None
                current_h = None
                ir = -1  # R index

                for _ in range(nrpts * num_wann * num_wann):
                    parts = f.readline().split()
                    if len(parts) < 7:
                        raise RuntimeError("Bad line in hr.dat (too few columns).")
                    r1, r2, r3 = map(int, parts[:3])
                    m, n = map(int, parts[3:5])  # 1-based indices
                    re, im = map(float, parts[5:7])

                    r = (r1, r2, r3)

                    # Start new block when encountering a new R vector
                    if current_r is None or r != current_r:
                        if current_r is not None:
                            # Save previous block
                            r_list.append(current_r)
                            h_list.append(current_h)
                        current_r = r
                        current_h = np.zeros((num_wann, num_wann), dtype=complex)
                        ir += 1

                    # Store matrix element (convert to 0-based indices)
                    i = m - 1
                    j = n - 1
                    current_h[i, j] = re + 1j * im

                # Save final block
                if current_r is not None:
                    r_list.append(current_r)
                    h_list.append(current_h)

            # Sanity check: number of R vectors should match header
            if len(r_list) != nrpts:
                raise RuntimeError(
                    f"nrpts mismatch: header {nrpts}, parsed {len(r_list)}"
                )

            self.num_wann = num_wann
            self.r_list = np.array(r_list, dtype=int)
            self.h_list = np.array(h_list, dtype=complex)
            self.ndegen = ndegen

            print(f"  Number of Wannier functions: {num_wann}")
            print(f"  Number of R vectors: {nrpts}")
            print(f"  GPU acceleration: {'Enabled' if self.use_gpu else 'Disabled'}")

            # If folder and seedname are not set, try to infer them from filename
            if self.folder is None or self.seedname is None:
                # Try to extract seedname from filename (e.g., "system_hr.dat" -> "system")
                basename = os.path.basename(filename)
                if basename.endswith("_hr.dat"):
                    inferred_seedname = basename[:-7]  # Remove "_hr.dat"
                    inferred_folder = os.path.dirname(filename)
                    print(f"  Inferred seedname: {inferred_seedname}")
                    print(f"  Inferred folder: {inferred_folder}")

        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {filename}") from e

    def _initialize_gpu_arrays(self) -> None:
        """
        Initialize GPU arrays if not already done.
        This is called automatically when GPU methods are used.
        """
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy is not available. Cannot use GPU acceleration.")

        if self.r_list_gpu is None and self.r_list is not None:
            self.r_list_gpu = cp.asarray(self.r_list)
            self.h_list_gpu = cp.asarray(self.h_list)
            self.ndegen_gpu = cp.asarray(self.ndegen, dtype=cp.float64)

    def hk(self, k_frac: tuple[float, float, float] | np.ndarray) -> np.ndarray:
        """
        Compute Bloch Hamiltonian H(k) from real-space Hamiltonian H(R).

        Parameters
        ----------
        k_frac : tuple or array-like, shape (3,)
            Fractional k-point coordinates in reciprocal lattice units.

        Returns
        -------
        hk_matrix : ndarray, shape (num_wann, num_wann)
            Complex Bloch Hamiltonian H(k).

        Raises
        ------
        ValueError
            If Hamiltonian data is not loaded.
        """
        if self.h_list is None:
            raise ValueError(
                "Hamiltonian data not loaded. Call load_hr_file() or load_from_seedname() first."
            )

        # Use GPU version if available and enabled
        if self.use_gpu:
            return self.hk_cuda(k_frac)

        # Original CPU implementation
        k = np.array(k_frac, dtype=float)
        nrpts, num_wann = self.h_list.shape[0], self.h_list.shape[1]

        # Phase factor: exp[2πi k·R]
        phases = np.exp(2j * np.pi * (self.r_list @ k))  # shape (nrpts,)

        # Apply degeneracy weighting (following Wannier90 User Guide)
        weights = phases / self.ndegen  # broadcasting: shape (nrpts,)

        # Weighted sum: Σ_R H(R) * weight(R)
        hk_matrix = np.zeros((num_wann, num_wann), dtype=complex)
        for ir in range(nrpts):
            hk_matrix += self.h_list[ir] * weights[ir]

        return hk_matrix

    def hk_cuda(
        self, k_frac: tuple[float, float, float] | np.ndarray, return_gpu: bool = False
    ) -> np.ndarray | cp.ndarray:
        """
        Compute Bloch Hamiltonian H(k) using CUDA acceleration.

        Parameters
        ----------
        k_frac : tuple or array-like, shape (3,)
            Fractional k-point coordinates in reciprocal lattice units.
        return_gpu : bool, optional
            If True, return CuPy array (stays on GPU); if False, return numpy array.
            Default is False.

        Returns
        -------
        hk_matrix : ndarray or cp.ndarray, shape (num_wann, num_wann)
            Complex Bloch Hamiltonian H(k).

        Raises
        ------
        ValueError
            If Hamiltonian data is not loaded.
        RuntimeError
            If GPU acceleration is not available.
        """
        if self.h_list is None:
            raise ValueError(
                "Hamiltonian data not loaded. Call load_hr_file() or load_from_seedname() first."
            )

        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy is not available. Cannot use GPU acceleration.")

        # Initialize GPU arrays if needed
        self._initialize_gpu_arrays()

        # Convert k to GPU array
        k = cp.asarray(k_frac, dtype=cp.float64)
        nrpts, num_wann = self.h_list_gpu.shape[0], self.h_list_gpu.shape[1]

        # Phase factor: exp[2πi k·R] on GPU
        phases = cp.exp(2j * cp.pi * (self.r_list_gpu @ k))  # shape (nrpts,)

        # Apply degeneracy weighting
        weights = phases / self.ndegen_gpu  # broadcasting: shape (nrpts,)

        # Weighted sum: Σ_R H(R) * weight(R) on GPU
        hk_matrix_gpu = cp.zeros((num_wann, num_wann), dtype=cp.complex128)
        for ir in range(nrpts):
            hk_matrix_gpu += self.h_list_gpu[ir] * weights[ir]

        # Return GPU array if requested
        if return_gpu:
            return hk_matrix_gpu

        # Convert back to numpy array
        return cp.asnumpy(hk_matrix_gpu)

    def load_wout_file(self, filename: str) -> dict:
        """
        Load and parse data from Wannier90 .wout file.

        Parameters
        ----------
        filename : str
            Path to Wannier90 .wout file

        Returns
        -------
        dict
            Dictionary containing parsed data:
            - 'disentangle_data': (3, num_iter) array with [iteration, time, delta]
            - 'wannierise_spreads': (num_cycle, num_wann, 1) array with spreads
            - 'wannierise_od': (num_cycle, 1) array with O_D values
            - 'disentangle_tar': float, convergence tolerance for disentanglement
            - 'wannierise_tar': float, convergence tolerance for wannierisation
        """
        data_iter = []
        data_delta = []
        data_time = []

        # Wannierise data containers
        num_wann = 0
        spreads_data = []  # Each element is a list of spreads for one cycle
        od_data = []  # O_D value for each cycle
        current_cycle = -1  # -1 represents Initial State, 0+ represents cycles

        # Convergence tolerance values
        disentangle_tar = None
        wannierise_tar = None

        # Flags for parsing state
        in_dis_section = False
        in_dis_header = False  # Flag for DISENTANGLE header section
        in_wannierise_header = False  # Flag for WANNIERISE header section
        header_found = False
        found_first_wannierise = False
        in_wannierise_section = False
        reading_spreads = False
        current_spreads = []

        # First pass: get the number of Wannier functions
        with open(filename, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "Number of Wannier Functions" in line:
                    match = re.search(r":\s*(\d+)", line)
                    if match:
                        num_wann = int(match.group(1))
                        break

        # Second pass: parse all data in a single file read
        with open(filename, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()

                # Check for DISENTANGLE header section
                if (
                    "*------------------------------- DISENTANGLE --------------------------------*"
                    in line
                ):
                    in_dis_header = True
                elif in_dis_header:
                    if (
                        "*----------------------------------------------------------------------------*"
                        in line
                    ):
                        in_dis_header = False
                    elif "Convergence tolerence" in line:
                        # Extract tolerance value from line like:
                        # "|  Convergence tolerence                     :         1.000E-08             |"
                        match = re.search(r":\s*([\d.E+-]+)", line)
                        if match:
                            try:
                                disentangle_tar = float(match.group(1))
                            except ValueError:
                                disentangle_tar = None

                # Check for WANNIERISE header section
                if (
                    "*------------------------------- WANNIERISE ---------------------------------*"
                    in line
                ):
                    in_wannierise_header = True
                elif in_wannierise_header:
                    if (
                        "*----------------------------------------------------------------------------*"
                        in line
                    ):
                        in_wannierise_header = False
                    elif "Convergence tolerence" in line:
                        # Extract tolerance value from line like:
                        # "|  Convergence tolerence                     :         0.200E-09             |"
                        match = re.search(r":\s*([\d.E+-]+)", line)
                        if match:
                            try:
                                wannierise_tar = float(match.group(1))
                            except ValueError:
                                wannierise_tar = None

                # Disentangle section parsing (existing code)
                if not in_dis_section:
                    # Look for "Extraction of optimally-connected subspace" as start marker
                    if "Extraction of optimally-connected subspace" in line:
                        in_dis_section = True
                elif in_dis_section:
                    # Stop extraction when "Final Omega_I" is encountered
                    if "Final Omega_I" in line:
                        in_dis_section = False

                    # Skip header row
                    if (
                        not header_found
                        and line.startswith("|")
                        and "Iter" in line
                        and "Time" in line
                    ):
                        header_found = True

                    if (
                        line.startswith("+---")
                        or line.startswith("+---")
                        or line.startswith("+---")
                    ):
                        continue

                    # Continue reading if convergence marker is found until Final Omega_I
                    if line.startswith("<<"):
                        continue

                    # Parse data rows
                    # Example format: "      1     413.16891841     405.30583437       1.940E-02      0.00    <-- DIS"
                    match = re.match(
                        r"^\s*(\d+)\s+[\d.E+-]+\s+[\d.E+-]+\s+([\d.E+-]+)\s+([\d.E+-]+)",
                        line,
                    )
                    if match:
                        iter_num = int(match.group(1))
                        delta = float(match.group(2))
                        time = float(match.group(3))
                        data_iter.append(iter_num)
                        data_delta.append(delta)
                        data_time.append(time)

                # Wannierise section parsing (existing code)
                # Look for WANNIERISE section
                if (
                    "*------------------------------- WANNIERISE ---------------------------------*"
                    in line
                ):
                    if not found_first_wannierise:
                        # Found the first WANNIERISE section (parameter section), skip it
                        found_first_wannierise = True
                    else:
                        # Found the second WANNIERISE section (calculation section), start reading
                        in_wannierise_section = True

                # Old detection method (kept for compatibility)
                if "*--- WANNIERISE ---*" in line and not found_first_wannierise:
                    found_first_wannierise = True
                elif "*--- WANNIERISE ---*" in line and found_first_wannierise:
                    in_wannierise_section = True

                if in_wannierise_section:
                    # Check if WANNIERISE section has ended
                    if (
                        "*---" in line and "---*" in line and "WANNIERISE" not in line
                    ) or "All done" in line:
                        # Save the last cycle's data (if any)
                        if current_spreads and len(current_spreads) == num_wann:
                            spreads_data.append(current_spreads)
                            current_spreads = []
                        in_wannierise_section = False

                    # Identify cycle start
                    if line.startswith("Initial State"):
                        current_cycle = -1
                        reading_spreads = True
                        current_spreads = []

                    # Identify cycle number
                    if line.startswith("Cycle:"):
                        # Save previous cycle's data
                        if current_spreads and len(current_spreads) == num_wann:
                            spreads_data.append(current_spreads)
                            current_spreads = []

                        # Extract cycle number
                        match = re.search(r"Cycle:\s+(\d+)", line)
                        if match:
                            current_cycle = int(match.group(1))
                        reading_spreads = True

                    # Read WF centre and spread lines
                    if reading_spreads and line.startswith("WF centre and spread"):
                        # Example: "WF centre and spread    1  (  1.985112,  3.438416,100.291488 )     6.75594284"
                        match = re.search(
                            r"WF centre and spread\s+\d+\s+\([^)]+\)\s+([\d.]+)", line
                        )
                        if match:
                            spread = float(match.group(1))
                            current_spreads.append(spread)

                    # End of spread section when encountering "Sum of centres and spreads"
                    if "Sum of centres and spreads" in line:
                        reading_spreads = False

                    # Read O_D value - only read O_D from DLTA lines
                    if "O_D=" in line and "<-- DLTA" in line:
                        match = re.search(r"O_D=\s*([\d.E+-]+)", line)
                        if match:
                            od = float(match.group(1))
                            od_data.append(od)

        # Process the last cycle's data for wannierise
        if current_spreads and len(current_spreads) == num_wann:
            spreads_data.append(current_spreads)

        # Convert disentangle data using ternary operator
        dis_data = (
            np.array([data_iter, data_time, data_delta]) if data_iter else np.array([])
        )

        # Convert wannierise data
        if spreads_data and num_wann > 0:
            # Check if there's an Initial State (cycle=-1)
            # If so, start from the first actual cycle (remove Initial State)
            if len(spreads_data) > 1 and current_cycle == -1:
                spreads_data = spreads_data[1:]  # Remove Initial State

            # Similarly process od_data
            if len(od_data) > 1:
                od_data = od_data[1:]  # Remove O_D corresponding to Initial State

            # Convert to numpy arrays
            spreads_array = np.array(spreads_data).reshape(-1, num_wann, 1)
            od_array = np.array(od_data).reshape(-1, 1)
        else:
            spreads_array = np.array([])
            od_array = np.array([])

        return {
            "disentangle_data": dis_data,
            "wannierise_spreads": spreads_array,
            "wannierise_od": od_array,
            "disentangle_tar": disentangle_tar,
            "wannierise_tar": wannierise_tar,
        }

    def load_reciprocal_vectors(self, filename: str) -> np.ndarray:
        """
        Read reciprocal lattice vectors from Wannier90 .wout file or OpenMX .out file.

        Parameters
        ----------
        filename : str
            Path to Wannier90 .wout file or OpenMX .out file.

        Returns
        -------
        bvecs : (3,3) ndarray
            Reciprocal lattice vectors b1, b2, b3 in 1/Ångström.
            Returns None if not found.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        """
        b1 = b2 = b3 = None

        # Determine file type based on extension
        file_ext = os.path.splitext(filename)[1].lower()

        try:
            with open(filename, encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

                for i, line in enumerate(lines):
                    # For .wout files (Wannier90 format)
                    if file_ext == ".wout":
                        if (
                            "Reciprocal-Space Vectors" in line
                            or "Reciprocal Vectors" in line
                        ):
                            # Read next 3 lines for b1, b2, b3
                            for j in range(1, 4):
                                if i + j < len(lines):
                                    b_line = lines[i + j]
                                    if "b_1" in b_line:
                                        b1 = self._parse_wannier_vector_line(b_line)
                                    elif "b_2" in b_line:
                                        b2 = self._parse_wannier_vector_line(b_line)
                                    elif "b_3" in b_line:
                                        b3 = self._parse_wannier_vector_line(b_line)
                            break

                    # For .out files (OpenMX format)
                    elif file_ext == ".out":
                        if "Reciprocal vector b1" in line:
                            b1 = self._parse_openmx_vector_line(line)
                        elif "Reciprocal vector b2" in line:
                            b2 = self._parse_openmx_vector_line(line)
                        elif "Reciprocal vector b3" in line:
                            b3 = self._parse_openmx_vector_line(line)
                            # If we found all three vectors, break
                            if b1 is not None and b2 is not None and b3 is not None:
                                break

        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {filename}") from e

        if b1 is None or b2 is None or b3 is None:
            return None

        return np.vstack([b1, b2, b3])

    @staticmethod
    def _parse_wannier_vector_line(line: str) -> np.ndarray:
        """Helper to extract a vector from a Wannier90 line like:
        'b_1     1.474634   0.851380   0.000000'
        """
        vals = line.split()
        return np.array(list(map(float, vals[1:4])))

    @staticmethod
    def _parse_openmx_vector_line(line: str) -> np.ndarray:
        """Helper to extract a vector from an OpenMX line like:
        '#  Reciprocal vector b1 (1/Ang):   2.55414  1.47463  0.00000'
        """
        right = line.split(":", 1)[1]
        vals = right.split()
        return np.array(list(map(float, vals[:3])))

    def _extract_disentangle_data(self, filename):
        """
        Extract Iter, Delta(frac.), Time data from the DISENTANGLE section in seedname.wout.

        Parameters
        ----------
        filename : str
            Path to the Wannier90 output file (.wout)

        Returns
        -------
        numpy.ndarray
            A (3, num_iter) numpy array containing iteration, time, and delta values,
            or an empty array if no data is found.
        """
        # This method is kept for backward compatibility but delegates to load_wout_file
        result = self.load_wout_file(filename)
        return result["disentangle_data"]

    def _extract_wannierise_data(self, filename):
        """
        Extract data from the WANNIERISE section in seedname.wout.

        Parameters
        ----------
        filename : str
            Path to the Wannier90 output file (.wout)

        Returns
        -------
        tuple
            - spreads_array: numpy array of shape (num_cycle, num_wann, 1) containing
              spread of each Wannier function in each cycle
            - od_array: numpy array of shape (num_cycle, 1) containing O_D values for each cycle
        """
        # This method is kept for backward compatibility but delegates to load_wout_file
        result = self.load_wout_file(filename)
        return result["wannierise_spreads"], result["wannierise_od"]

    def set_use_gpu(self, use_gpu: bool) -> None:
        """
        Enable or disable GPU acceleration.

        Parameters
        ----------
        use_gpu : bool
            Whether to use GPU acceleration.
        """
        if use_gpu and not CUPY_AVAILABLE:
            print("Warning: CuPy not available. GPU acceleration disabled.")
            self.use_gpu = False
        else:
            self.use_gpu = use_gpu

        # Clear GPU arrays if disabling GPU
        if not self.use_gpu:
            self.r_list_gpu = None
            self.h_list_gpu = None
            self.ndegen_gpu = None
