from pathlib import Path

import numpy as np

from stm_data_processing.utils.lattice_loader import BVecs

try:
    import cupy as cp

    CUPY_AVAILABLE = True
    print("CuPy is available. GPU acceleration enabled.")
    ArrayType = np.ndarray | cp.ndarray
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    print("CuPy not found. Using CPU-only mode.")
    ArrayType = np.ndarray


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

        self.bvecs = None
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
        hr_file = Path(folder) / f"{seedname}_hr.dat"

        # Check if hr.dat file exists
        if not hr_file.exists():
            raise FileNotFoundError(f"hr.dat file not found: {hr_file}")

        # Try to find std_file (wout or out file)
        wout_file = Path(folder) / f"{seedname}.wout"
        out_file = Path(folder) / f"{seedname}.out"

        std_file = None
        bvecs_obj = None
        if wout_file.exists():
            std_file = wout_file
            print(f"  Found wout file: {wout_file}")
            bvecs_obj = BVecs(std_file)

        elif out_file.exists():
            std_file = out_file
            print(f"  Found out file: {out_file}")
            bvecs_obj = BVecs(std_file)

        else:
            print(
                f"Warning: Neither {seedname}.wout nor {seedname}.out found in {folder}"
            )
        self.bvecs = bvecs_obj.bvecs if bvecs_obj is not None else None
        self.folder = folder
        self.seedname = seedname
        self.std_file = std_file

        # Load hr.dat file
        self.load_hr_file(hr_file)

        print(f"  Wannier90 seedname: {seedname}")
        print(f"  Folder: {folder}")
        print(f"  hr.dat file: {hr_file}")
        if std_file:
            print(f"  std_file (auto-detected): {std_file}")

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
            with Path(filename).open() as f:
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
                basename = Path(filename).name
                if basename.endswith("_hr.dat"):
                    # Remove "_hr.dat"
                    inferred_seedname = basename[:-7]
                    inferred_folder = str(Path(filename).parent)
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
    ) -> ArrayType:
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
