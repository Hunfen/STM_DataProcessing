import contextlib
import numpy as np
import os


def parse_dft_band_data(
    fname_band: str | None = None,
    folder: str | None = None,
    systemname: str | None = None,
) -> dict[str, np.ndarray | list]:
    """
    Parse band structure data from OpenMX .Band file.

    Parameters
    ----------
    fname_band : str, optional
        Path to the band data file (e.g., C6LiC6.Band). If None and folder/systemname
        are set during initialization, constructs path as folder/systemname.Band.
    folder : str, optional
        Folder containing OpenMX files.
    systemname : str, optional
        System name for OpenMX files.

    Returns
    -------
    dict
        Dictionary containing:
        - 'dist' : 1D array of cumulative k-path distance (1/Å)
        - 'bands' : (nk_total, nband) array of band energies (eV, EF=0)
        - 'tick_pos' : list of float, x-positions of high-symmetry points
        - 'tick_label' : list of str, labels for high-symmetry points
        - 'kpts_frac' : (nk_total, 3) fractional k coordinates
        - 'kpts_cart' : (nk_total, 3) cartesian k coordinates (1/Å)
        - 'fermi_energy' : Fermi energy in eV
        - 'n_bands' : Number of bands

    Raises
    ------
    ValueError
        If input parameters are invalid.
    FileNotFoundError
        If no valid band file can be determined or file is not found.
    """
    h2ev = 27.211386245988  # Hartree to eV
    au2ang = 1.8897261254578281  # Bohr radius to Angstrom conversion factor

    # Determine the band file path
    if fname_band is None:
        if folder is not None and systemname is not None:
            fname_band = os.path.join(folder, f"{systemname}.Band")
            print(f"Using auto-detected band file: {fname_band}")
        else:
            raise ValueError(
                "fname_band not provided and folder/systemname not set. "
                "Please provide fname_band or both folder and systemname."
            )
    else:
        print(f"Using provided band file: {fname_band}")

    # Read raw lines from .Band file
    try:
        with open(fname_band, encoding="ISO-8859-1") as f:
            raw = [line.strip() for line in f if line.strip()]
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Band file not found: {fname_band}") from e

    if not raw:
        raise ValueError(f"Empty band file: {fname_band}")

    # First line: number of bands and Fermi energy (in Hartree)
    header = raw[0].split()
    if len(header) < 3:
        raise ValueError(f"Invalid header in band file: {raw[0]}")

    nband = int(header[0])
    mu_au = float(header[2])
    fermi_energy = mu_au * h2ev

    print("Parsing band data:")
    print(f"  Number of bands: {nband}")
    print(f"  Fermi energy: {fermi_energy:.4f} eV")

    # Second line: Reciprocal lattice vectors in a.u.
    b_line = raw[1].split()
    if len(b_line) < 9:
        raise ValueError(f"Invalid reciprocal lattice vector line: {raw[1]}")

    b1 = np.array(list(map(float, b_line[0:3]))) * au2ang
    b2 = np.array(list(map(float, b_line[3:6]))) * au2ang
    b3 = np.array(list(map(float, b_line[6:9]))) * au2ang
    b = np.array([b1, b2, b3])

    # Third line: Number of k-paths
    n_kpaths = int(raw[2])

    # Read k-path information and build nk list and tick labels
    nk = []
    tick_labels = []  # Store all tick labels
    lines = raw[3:]
    i = 0

    for path_idx in range(n_kpaths):
        if i >= len(lines):
            raise ValueError("Unexpected end of file while reading k-path info")

        # Each k-path line has format:
        # Num_kpoints_on_path, start_frac_x, start_frac_y, start_frac_z, end_frac_x, end_frac_y, end_frac_z, label_start, label_end
        path_info = lines[i].split()
        if len(path_info) < 8:
            raise ValueError(f"Invalid k-path info line (missing labels): {lines[i]}")

        num_kpts = int(path_info[0])
        nk.append(num_kpts)

        # Extract labels (last two elements)
        start_label = path_info[-2]
        end_label = path_info[-1]

        # For the first path, add both start and end labels
        if path_idx == 0:
            tick_labels.append(start_label)
        # Always add the end label (this handles consecutive paths correctly)
        tick_labels.append(end_label)

        i += 1

    # Now process dispersion data
    kpts_frac = []
    bands = []

    while i < len(lines):
        toks = lines[i].split()

        # Skip lines that don't look like k-point headers
        if len(toks) < 4 or not toks[0].isdigit():
            i += 1
            continue

        n = int(toks[0])
        if n != nband:
            i += 1
            continue

        try:
            kx, ky, kz = map(float, toks[1:4])
        except ValueError:
            i += 1
            continue

        kpts_frac.append([kx, ky, kz])

        # Collect band energies
        e_vals_au = []
        j = i + 1
        while j < len(lines) and len(e_vals_au) < nband:
            for t in lines[j].split():
                with contextlib.suppress(ValueError):
                    e_vals_au.append(float(t))
            j += 1

        e_vals_au = np.array(e_vals_au[:nband])
        e_vals = (e_vals_au - mu_au) * h2ev
        bands.append(e_vals)
        i = j

    kpts_frac = np.array(kpts_frac)
    bands = np.array(bands)

    print(f"  Number of k-points: {len(kpts_frac)}")
    print(f"  Energy range (eV, EF=0): {bands.min():.4f} → {bands.max():.4f}")

    kpts_cart = kpts_frac @ b

    # Compute cumulative distances
    dist = np.zeros(len(kpts_cart))
    for j in range(1, len(kpts_cart)):
        dk = np.linalg.norm(kpts_cart[j] - kpts_cart[j - 1])
        dist[j] = dist[j - 1] + dk

    # Compute tick positions based on nk list
    idx = [0]
    cum = 0
    for n in nk:
        cum += n
        idx.append(cum - 1)
    tick_pos = [dist[k] for k in idx]

    # Verify that tick_pos and tick_labels have the same length
    if len(tick_pos) != len(tick_labels):
        raise RuntimeError(f"Mismatch between tick positions ({len(tick_pos)}) and labels ({len(tick_labels)})")

    return {
        "dist": dist,
        "bands": bands,
        "tick_pos": tick_pos,
        "tick_label": tick_labels,
        "kpts_frac": kpts_frac,
        "kpts_cart": kpts_cart,
        "fermi_energy": fermi_energy,
        "n_bands": nband,
    }


def openmx_band_analysis(
    band_file: str | None = None,
    folder: str | None = None,
    systemname: str | None = None,
) -> dict[str, np.ndarray | list]:
    """
    Parse OpenMX band structure data.

    Parameters
    ----------
    band_file : str, optional
        Path to OpenMX .Band file. If provided, this takes precedence.
    folder : str, optional
        Path to folder containing OpenMX files.
    systemname : str, optional
        System name for OpenMX files.

    Returns
    -------
    dict
        Parsed band structure data containing:
        - 'dist' : 1D array of cumulative k-path distance (1/Å)
        - 'bands' : (nk_total, nband) array of band energies (eV, EF=0)
        - 'tick_pos' : list of float, x-positions of high-symmetry points
        - 'tick_label' : list of str, labels for high-symmetry points
        - 'kpts_frac' : (nk_total, 3) fractional k coordinates
        - 'kpts_cart' : (nk_total, 3) cartesian k coordinates (1/Å)
        - 'fermi_energy' : Fermi energy in eV
        - 'n_bands' : Number of bands

    Notes
    -----
    Either provide band_file directly, or provide folder and systemname.
    If both are provided, band_file takes precedence.
    """
    if band_file is not None:
        # Use direct band file path
        return parse_dft_band_data(fname_band=band_file)
    elif folder is not None and systemname is not None:
        # Use folder and systemname
        return parse_dft_band_data(folder=folder, systemname=systemname)
    else:
        raise ValueError("Either provide band_file, or provide both folder and systemname.")
