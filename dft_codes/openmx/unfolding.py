import numpy as np
import pandas as pd
from .parser import OpenMX

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def _generate_orbital_column_names(openmx_parser: OpenMX) -> list[str]:
    """
    Generate column names for orbital weights based on atomic positions and species.

    Parameters
    ----------
    openmx_parser : OpenMX
        Initialized OpenMX parser instance with parsed atomic data

    Returns
    -------
    list[str]
        List of column names in format '<atom_index>-<element>-<orbital>'
    """
    # Ensure atomic positions data is available
    if openmx_parser.atomic_positions_data is None:
        raise RuntimeError("Atomic positions data not available. Call read_atomic_positions() first.")

    elements = openmx_parser.atomic_positions_data["elements"]

    # Create element to orbitals mapping from atomic_species
    element_to_orbitals = {}
    if openmx_parser.atomic_species is not None:
        for species in openmx_parser.atomic_species:
            element_to_orbitals[species["element"]] = species["orbitals"]
    else:
        raise RuntimeError("Atomic species data not available. Call read_atomic_species_from_out() first.")

    # D orbital names in order
    d_orbital_names = ["d3z^2-r^2", "dx^2-y^2", "dxy", "dxz", "dyz"]

    column_names = []

    for atom_idx, element in enumerate(elements):
        if element not in element_to_orbitals:
            raise ValueError(f"Element {element} not found in atomic species data")

        orbitals = element_to_orbitals[element]
        s_count = orbitals.get("s", 0)
        p_count = orbitals.get("p", 0)
        d_count = orbitals.get("d", 0)

        # Add s orbitals
        for s_idx in range(s_count):
            column_names.append(f"{atom_idx}-{element}-{s_idx}s")

        # Add p orbitals
        for p_idx in range(p_count):
            column_names.append(f"{atom_idx}-{element}-{p_idx}px")
            column_names.append(f"{atom_idx}-{element}-{p_idx}py")
            column_names.append(f"{atom_idx}-{element}-{p_idx}pz")

        # Add d orbitals
        for d_idx in range(d_count):
            for d_name in d_orbital_names:
                column_names.append(f"{atom_idx}-{element}-{d_idx}{d_name}")

    return column_names


def _get_target_weights_from_df(
    df: pd.DataFrame,
    element: str | None = None,
    atom_index: int | None = None,
) -> np.ndarray:
    """Internal helper to compute weights from df columns."""

    def match_col(col: str) -> bool:
        parts = col.split("-", 2)
        if len(parts) != 3:
            return False
        idx_str, elem, _ = parts
        try:
            idx = int(idx_str)
        except ValueError:
            return False
        if atom_index is not None and idx != atom_index:
            return False
        return element is None or elem == element

    weight_cols = [col for col in df.columns if match_col(col)]
    if not weight_cols:
        if element is None and atom_index is None:
            # Use all orbital columns
            weight_cols = [col for col in df.columns if col not in ("kpath", "energy")]
        else:
            raise ValueError(f"No columns match element={element}, atom_index={atom_index}")

    return df[weight_cols].sum(axis=1).values


def lorentzian_2d(k, e, k0, e0, delta_k, delta_e):
    """
    2D Lorentzian broadening kernel.

    Works with both NumPy and CuPy arrays due to API compatibility.
    """
    return 1.0 / (((k - k0) / delta_k) ** 2 + ((e - e0) / delta_e) ** 2 + 1.0)


def compute_spectral_function(
    df: pd.DataFrame,
    *,
    element: str | None = None,
    atom_index: int | None = None,
    nk: int = 512,
    ne: int = 512,
    delta_k_input_nm: float = 100,
    delta_e_input_k: float = 100,
    use_gpu: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the spectral function A(k, E) from OpenMX unfolding data.

    Parameters
    ----------
    df : pd.DataFrame
        Output from `read_unfold_orbup()`, containing 'kpath', 'energy', and orbital weights.
    element : str, optional
        Element symbol (e.g., 'C') to project onto.
    atom_index : int, optional
        Specific atom index as labeled in the column header (typically 0-based in OpenMX).
        If both `element` and `atom_index` are None, all orbitals are summed.
    nk, ne : int
        Number of grid points along k-path and energy axes.
        The range is automatically set to [min, max] of df['kpath'] and df['energy'].
    delta_k_input_nm : float
        Real-space coherence length L (nm). Momentum HWHM: δk = 2π / (L in Å).
    delta_e_input_k : float
        Temperature (K) for thermal broadening. Must satisfy 0.1 <= T <= 300.
        Energy HWHM is derived from the half-width of -∂f/∂E at this temperature.
    use_gpu : bool, optional
        Whether to use GPU acceleration if available (default: False).

    Returns
    -------
    k, e : np.ndarray
        Meshgrid arrays of shape (ne, nk).
    a : np.ndarray
        Spectral function A(k, E) of shape (ne, nk).
    """
    # --- Validate temperature ---
    if not (0.1 <= delta_e_input_k <= 300.0):
        raise ValueError(f"delta_e_input_k (temperature) must be between 0.1 K and 300 K. Got {delta_e_input_k} K.")

    # --- Compute weights based on selection ---
    w_array = _get_target_weights_from_df(df, element=element, atom_index=atom_index)

    # --- Automatically set plot range from data ---
    k_min, k_max = df["kpath"].min(), df["kpath"].max()
    e_min, e_max = df["energy"].min(), df["energy"].max()

    # --- Broadening parameters ---
    l_a = 10.0 * delta_k_input_nm  # nm → Å
    delta_k_angstrom_inv = (2.0 * np.pi) / l_a

    kb_ev_per_k = 8.617333262145e-5
    x0 = np.log(3 + 2 * np.sqrt(2))  # ≈1.7627
    delta_e_ev = x0 * kb_ev_per_k * delta_e_input_k

    # --- Create grid ---
    k_vals = np.linspace(k_min, k_max, nk)
    e_vals = np.linspace(e_min, e_max, ne)

    if use_gpu and CUPY_AVAILABLE:
        # GPU version - simple loop but efficient
        k_gpu = cp.asarray(k_vals)
        e_gpu = cp.asarray(e_vals)
        k_grid, e_grid = cp.meshgrid(k_gpu, e_gpu, indexing="ij")
        a_gpu = cp.zeros_like(k_grid)

        # Transfer data to GPU once
        kpath_gpu = cp.asarray(df["kpath"].values)
        energy_gpu = cp.asarray(df["energy"].values)
        w_array_gpu = cp.asarray(w_array)

        # Simple loop - each iteration processes entire grid on GPU
        for i in range(len(df)):
            k0 = kpath_gpu[i]
            e0 = energy_gpu[i]
            w = w_array_gpu[i]
            # This operation is vectorized over the entire (ne, nk) grid
            a_gpu += w * lorentzian_2d(k_grid, e_grid, k0, e0, delta_k_angstrom_inv, delta_e_ev)

        k, e, a = cp.asnumpy(k_grid), cp.asnumpy(e_grid), cp.asnumpy(a_gpu)

    else:
        # CPU version - simple loop
        k_grid, e_grid = np.meshgrid(k_vals, e_vals, indexing="ij")
        a = np.zeros_like(k_grid)

        for i in range(len(df)):
            k0 = df["kpath"].iloc[i]
            e0 = df["energy"].iloc[i]
            w = w_array[i]
            a += w * lorentzian_2d(k_grid, e_grid, k0, e0, delta_k_angstrom_inv, delta_e_ev)

    return k, e, a


def read_unfold_orbup(file_path: str, openmx_parser: OpenMX) -> pd.DataFrame:
    """
    Read and process OpenMX unfolding weight file (.unfold_orbup).

    This function reads the unfolding weight file and processes it to:
    1. Convert k-path from Bohr^-1 to Å^-1
    2. Keep energy as-is (already in eV relative to Fermi level, i.e., E_F = 0)
    3. Generate column names for orbital weights based on atomic positions and species

    Parameters
    ----------
    file_path : str
        Path to the .unfold_orbup file
    openmx_parser : OpenMX
        Initialized OpenMX parser instance with parsed atomic data

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'kpath': k-path values in Å^-1
        - 'energy': energy values in eV relative to Fermi level (E_F = 0)
        - Additional columns for each orbital weight with names like '0-C-s', '1-Li-px', etc.

    Notes
    -----
    Column naming convention: <atom_index>-<element>-<orbital>, where atom_index is 0-based,
    following Python indexing.
    The energy is assumed to already be referenced to the Fermi level (E - E_F).
    Orbital naming convention:
    - s orbitals: 0s, 1s, ..., (n-1)s
    - p orbitals: 0px, 0py, 0pz, 1px, 1py, 1pz, ..., (n-1)px, (n-1)py, (n-1)pz
    - d orbitals: 0d3z^2-r^2, 0dx^2-y^2, 0dxy, 0dxz, 0dyz, ..., (n-1)dyz
    """

    # Read the file
    data = np.loadtxt(file_path)

    # Extract kpath and energy columns
    kpath_bohr = data[:, 0]  # in Bohr^-1
    energy_ev = data[:, 1]  # already in eV and relative to E_F

    # Convert k-path: 1/Bohr -> 1/Å
    bohr_to_angstrom = 0.52917721092
    kpath_angstrom = kpath_bohr / bohr_to_angstrom

    # Get orbital weights (all columns from index 2 onwards)
    orbital_weights = data[:, 2:]

    # Generate column names for orbitals
    orbital_columns = _generate_orbital_column_names(openmx_parser)

    # Validate that the number of columns matches
    if orbital_weights.shape[1] != len(orbital_columns):
        raise ValueError(
            f"Number of orbital weight columns ({orbital_weights.shape[1]}) "
            f"does not match expected number from atomic data ({len(orbital_columns)})"
        )

    # Create DataFrame
    df_dict = {"kpath": kpath_angstrom, "energy": energy_ev}

    # Add orbital weight columns
    for i, col_name in enumerate(orbital_columns):
        df_dict[col_name] = orbital_weights[:, i]

    return pd.DataFrame(df_dict)
