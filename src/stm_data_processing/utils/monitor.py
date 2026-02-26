import re
from pathlib import Path

import numpy as np


def load_wout_file(filename: str) -> dict:
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
    with Path(filename).open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "Number of Wannier Functions" in line:
                match = re.search(r":\s*(\d+)", line)
                if match:
                    num_wann = int(match.group(1))
                    break

    # Second pass: parse all data in a single file read
    with Path(filename).open(encoding="utf-8", errors="ignore") as f:
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
