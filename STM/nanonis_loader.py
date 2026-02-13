"""Python module that helps read the Nanonis files."""

import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import path


class NanonisFileLoader:
    """Loader for Nanonis spectroscopy and scanning probe microscopy files.

    Supports three file formats:
    - .sxm: Scanning probe microscopy images
    - .dat: Spectroscopy curves and tabular data
    - .3ds: 3D spectroscopy grids (dI/dV maps, etc.)

    Implements lazy loading - raw file data is read during initialization,
    but header and data parsing only occurs when first accessed via properties.

    Example:
        >>> loader = NanonisFileLoader("data.sxm")
        >>> print(loader.header["SCAN_RANGE"])  # Triggers header parsing
        >>> img = loader.data  # Triggers data parsing

    """

    def __init__(self, f_path: str) -> None:
        """Initialize NanonisLoader with a file path.

        Args:
            f_path (str): path to the Nanonis file (.sxm, .dat, or .3ds)

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file is empty or has an unsupported file type

        """
        file_path_obj = path(f_path)
        if not file_path_obj.exists():
            error_msg = f"File not found: {f_path}"
            raise FileNotFoundError(error_msg)
        file_size = file_path_obj.stat().st_size
        if file_size == 0:
            error_msg = f"File is empty: {f_path}"
            raise ValueError(error_msg)

        self.file_path, self.fname = os.path.split(f_path)
        self._raw_header = None
        self._raw_data = None
        self._header = None
        self._data = None
        self._parameters = None
        self._pixels = None

        if f_path.endswith(".sxm"):
            self.file_type = "sxm"
            self._raw_header, self._raw_data = self._sxm_loader(f_path)
        elif f_path.endswith(".dat"):
            self.file_type = "dat"
            self._raw_header, self._raw_data = self._dat_loader(f_path)
        elif f_path.endswith(".3ds"):
            self.file_type = "3ds"
            self._raw_header, self._raw_data = self._3ds_loader(f_path)
            error_msg = f"Unsupported file type: {f_path}"
            raise ValueError(error_msg)

    def _sxm_loader(self, f_path: str) -> tuple[dict, np.ndarray]:
        """Load and parse .sxm file, extracting header and binary data.

        Reads header until ":SCANIT_END:" marker, then extracts binary
        data as big-endian floats. Header entries start with ":[key]:"
        followed by content lines.

        Args:
            f_path: path to .sxm file as string.

        Returns:
            Tuple containing:
                - raw_header: Dictionary of parsed header entries
                - data: 1D numpy array of binary data

        Notes:
            - Header entries must start with ":[key]:" markers
            - Skips 2 lines after header before reading binary data
            - Binary data read as big-endian 32-bit floats
            - Uses UTF-8 decoding with error replacement

        """
        raw_header = {}
        entry, contents = "", ""

        with path.open(f_path, "rb") as f:
            # Read and parse the header
            for line in f:
                decoded_line = line.decode(encoding="utf-8", errors="replace")
                if ":SCANIT_END:" in decoded_line:
                    # Add the last entry if it exists
                    if entry:
                        raw_header[entry] = contents.rstrip("\n")
                    break

                if re.fullmatch(r":.*:\n", decoded_line):
                    # This is a key line
                    if entry:
                        # Add previous entry
                        raw_header[entry] = contents.rstrip("\n")
                    # Start new entry
                    entry = decoded_line.rstrip("\n").strip(":")
                    contents = ""
                else:
                    # This is a content line
                    contents += decoded_line

            # Skip two lines after header and move file pointer
            for _ in range(2):
                f.readline()
            f.seek(2, 1)

            # Read the binary data
            data = np.fromfile(f, dtype=">f")

        return raw_header, data

    # def _reform_sxm_header(self) -> dict:
    #     """Reformats the raw header data from a .sxm file into a structured dictionary.

    #     This method processes the raw header data, which is initially stored
    #     as a dictionary of key-value pairs, and organizes it into a more
    #     structured format. It handles both external VI attributes and internal
    #     modules, and converts specific sections (e.g., "Z-CONTROLLER" and
    #     "DATA_INFO") into pandas DataFrames for easier manipulation.

    #     Returns:
    #         dict: A dictionary containing the reformatted header data.
    #         The dictionary includes:
    #             - Key-value pairs for non-module-specific attributes.
    #             - DataFrames for sections like "Z-CONTROLLER" and "DATA_INFO".
    #             - Nested dictionaries or DataFrames for module-specific
    #             attributes, depending on the number of attributes per module.

    #     """
    #     header = {}
    #     modules = []

    #     for key, value in self._raw_header.items():
    #         if re.search(">", key):  # initial dicts for modules
    #             if key.startswith("Ext. VI 1>"):
    #                 # deal with external VI attributes
    #                 # Use lstrip instead of strip for multi-character prefix
    #                 module_name = key[len("Ext. VI 1>") :]
    #                 if ">" in module_name:
    #                     modules.append(module_name.split(">")[0])
    #                 else:
    #                     modules.append(module_name)
    #             else:  # deal with the internal modules
    #                 modules.append(key.split(">")[0])
    #         else:
    #             header.update({key: value})

    #     # count the number of attributes in every module
    #     modules = Counter(modules)

    #     for key in ["Z-CONTROLLER", "DATA_INFO"]:
    #         df = pd.DataFrame(
    #             [row.split("\t") for row in self._raw_header[key].split("\n")],
    #         )
    #         df.columns = df.iloc[0]
    #         header[key] = df[1:].reset_index(drop=True).dropna(how="any")

    #     for module, count in modules.items():
    #         if count == 1:
    #             for key, value in self._raw_header.items():
    #                 # Handle the strip issue properly
    #                 if key.startswith("Ext. VI 1>"):
    #                     check_key = key[len("Ext. VI 1>") :]
    #                 else:
    #                     check_key = key

    #                 if check_key.startswith(module):
    #                     header.update({module: value.strip("\n")})
    #         else:
    #             header[module] = {}
    #             for key, value in self._raw_header.items():
    #                 # Handle the strip issue properly
    #                 if key.startswith("Ext. VI 1>"):
    #                     check_key = key[len("Ext. VI 1>") :]
    #                 else:
    #                     check_key = key

    #                 if check_key.startswith(module):
    #                     header[module].update({key.split(">")[-1]: value.strip("\n")})
    #                 else:
    #                     continue

    #     return header

    def _reform_sxm_header(self) -> dict:
        """Reformats the raw header data from a .sxm file into a structured dictionary.

        This method processes the raw header data, which is initially stored
        as a dictionary of key-value pairs, and organizes it into a more
        structured format. It handles both external VI attributes and internal
        modules, and converts specific sections (e.g., "Z-CONTROLLER" and
        "DATA_INFO") into pandas DataFrames for easier manipulation.

        Returns:
            dict: A dictionary containing the reformatted header data.
            The dictionary includes:
                - Key-value pairs for non-module-specific attributes.
                - DataFrames for sections like "Z-CONTROLLER" and "DATA_INFO".
                - Nested dictionaries or DataFrames for module-specific
                attributes, depending on the number of attributes per module.

        """
        header = {}
        modules = []

        # Extract modules and non-module entries
        for key, value in self._raw_header.items():
            if re.search(">", key):
                module_name = self._extract_ext_vi(key)
                modules.append(module_name)
            else:
                header.update({key: value})

        # Process special DataFrame sections
        for key in ["Z-CONTROLLER", "DATA_INFO"]:
            if key in self._raw_header:
                df = pd.DataFrame(
                    [row.split("\t") for row in self._raw_header[key].split("\n")],
                )
                df.columns = df.iloc[0]
                header[key] = df[1:].reset_index(drop=True).dropna(how="any")

        # Process modules
        modules = Counter(modules)
        self._process_modules(header, modules)

        return header

    def _reform_sxm_data(self) -> np.ndarray:
        """Reformats raw SXM data into structured array.

        Note: This method should only be called after header is parsed.
        """
        self._ensure_header_parsed()

        if "SCAN_PIXELS" in self._header:
            pixels = tuple(map(int, self._header["SCAN_PIXELS"].split()))
        else:
            pixels = (0, 0)

        if "DATA_INFO" in self._header:
            channels = self._header["DATA_INFO"]["Name"].tolist()
        else:
            channels = []

        dir_flag = self._header.get("SCAN_DIR") == "up"

        total_pts = len(channels) * 2 * pixels[0] * pixels[1]
        raw_data = self._raw_data

        if total_pts > raw_data.size:
            raw_data = np.concatenate(
                [raw_data, np.full(total_pts - raw_data.size, np.nan)],
            )

        data = raw_data[:total_pts].reshape((len(channels) * 2, *pixels))
        for i in range(data.shape[0]):
            if i % 2 != 0:
                data[i] = np.fliplr(data[i])
            if dir_flag:
                data[i] = np.flipud(data[i])
        return data

    def _dat_loader(self, f_path: str) -> tuple[dict, pd.DataFrame]:
        r"""Load Nanonis .dat file, parsing header and data sections.

        Reads binary file to handle line endings precisely. Header entries
        use format: key<tab>value<tab>\r\n. Data section starts after
        [DATA] marker with tab-delimited columns.

        Args:
            f_path (str): path to .dat file

        Returns:
            tuple: (raw_header, df) where:
                - raw_header (dict): Header key-value pairs
                - df (pd.DataFrame): Data table with auto-converted dtypes

        Note:
            - Preserves original line endings during header parsing
            - Converts numeric columns automatically
            - Handles malformed data with error replacement

        """
        raw_header = {}
        key, value_buffer = "", ""
        columns, data_lines = [], []

        with path.open(f_path, "rb") as f:
            for line in f:
                decoded_line = line.decode("utf-8", errors="replace")
                if "[DATA]" in decoded_line:  # End of header
                    columns = (
                        next(f)
                        .decode("utf-8", errors="replace")
                        .strip("\r\n")
                        .split("\t")
                    )
                    break
                if decoded_line.endswith("\t\r\n"):
                    # Update key and value when encounter '\t\r\n'
                    if "\t" in decoded_line.rstrip("\t\r\n"):
                        key, value_buffer = decoded_line.rstrip("\t\r\n").split("\t", 1)
                    else:
                        value_buffer += decoded_line.rstrip("\t\r\n")
                    raw_header.update({key: value_buffer})
                    value_buffer = ""
                elif "\t" in decoded_line:
                    key, value_buffer = decoded_line.split("\t", 1)
                else:
                    value_buffer += decoded_line
            for line in f:
                decoded_line = line.decode("utf-8", errors="replace")
                if decoded_line.rstrip("\r\n"):
                    data_lines.append(decoded_line.rstrip("\r\n").split("\t"))
        data = pd.DataFrame(data_lines, columns=columns).apply(
            pd.to_numeric,
            errors="coerce",
        )
        return raw_header, data

    def _reform_dat_header(self) -> dict:
        """Reformat the raw header data from a .dat file into a structured dictionary.

        This method processes the raw header data, which is initially stored
        as a dictionary of key-value pairs, and organizes it into a more
        structured format. It handles both external VI attributes and internal
        modules, and converts specific sections (e.g., "Bias Spectroscopy")
        into pandas DataFrames for easier manipulation.

        Returns:
            dict: A dictionary containing the reformatted header data.
            The dictionary includes:
                - Key-value pairs for non-module-specific attributes.
                - Nested dictionaries or DataFrames for module-specific
                attributes, depending on the number of attributes per module.

        """
        header = {}
        modules = []

        for key, value in self._raw_header.items():
            if re.search(">", key):  # initial dicts for modules
                if key.startswith("Ext. VI 1>"):
                    # deal with external VI attributes
                    module_name = key[len("Ext. VI 1>") :]
                    if ">" in module_name:
                        modules.append(module_name.split(">")[0])
                    else:
                        modules.append(module_name)
                else:  # deal with the internal modules
                    modules.append(key.split(">")[0])
            else:
                header.update({key: value.strip("\n")})

        modules = Counter(modules)

        for module, count in modules.items():
            if count == 1:
                for key, value in self._raw_header.items():
                    # Handle the strip issue properly
                    if key.startswith("Ext. VI 1>"):
                        check_key = key[len("Ext. VI 1>") :]
                    else:
                        check_key = key

                    if module == check_key.split(">")[0]:
                        header.update({module: value.strip("\r\n")})
            else:
                header[module] = {}
                for key, value in self._raw_header.items():
                    # Handle the strip issue properly
                    if key.startswith("Ext. VI 1>"):
                        check_key = key[len("Ext. VI 1>") :]
                    else:
                        check_key = key

                    if module == check_key.split(">")[0]:
                        header[module].update(
                            {key.split(">")[-1]: value.strip("\r\n").strip('"')},
                        )

        for key, value in header["Bias Spectroscopy"].items():
            if "MultiLine Settings" in key:
                header["Bias Spectroscopy"].update(
                    {
                        "MultiLine Settings": pd.DataFrame(
                            np.array(
                                [
                                    list(map(float, row.split(",")))
                                    for row in value.split(";")
                                ],
                            ),
                            columns=key.split(":")[-1].split(","),
                        ),
                    },
                )
                header["Bias Spectroscopy"].pop(key)
                break
        return header

    def _3ds_loader(self, f_path: str) -> tuple[dict, np.ndarray]:
        """Load Nanonis .3ds file, parsing header and binary data.

        Reads file in binary mode to handle header and data precisely.
        Header ends at ':HEADER_END:' marker. Data is read as big-endian
        floats after header.

        Args:
            f_path (str): path to .3ds file

        Returns:
            tuple: (raw_header, data_1d) where:
                - raw_header (dict): Header key-value pairs
                - data_1d (np.ndarray): 1D array of binary data

        Note:
            - Uses UTF-8 decoding with error replacement
            - Handles multi-line header values
            - Skips CRLF after header end marker

        """
        raw_header = {}
        key, value_buffer = "", ""
        with path.open(f_path, "rb") as f:
            for line in f:
                decoded_line = line.decode(encoding="utf-8", errors="replace")
                if ":HEADER_END:" in decoded_line:
                    # b'\x0d\x0a' is following the ":HEADER_END:"
                    # f.readline() will automaticly skip them
                    break
                if decoded_line.endswith("\r\n"):
                    # Update key and value when encounter '\r\n'
                    if "=" in decoded_line.strip("\r\n"):
                        key, value_buffer = decoded_line.strip("\r\n").split("=", 1)
                    else:
                        value_buffer += decoded_line.rstrip("\t\r\n")
                    raw_header.update({key: value_buffer})
                    value_buffer = ""
                elif "=" in decoded_line:
                    key, value_buffer = decoded_line.split("=", 1)
                else:
                    value_buffer += decoded_line
            data_1d = np.fromfile(f, dtype=">f")
        return raw_header, data_1d

    def _reform_3ds_header(self) -> dict:
        """Reformats the raw header data into a structured dictionary.

        Parses the raw header, organizes module-specific data, and converts
        specific fields (e.g., 'MultiLine Settings') into appropriate formats.

        Returns:
            dict: Structured header with module-specific data.

        """
        header = {}
        modules = []
        for key, value in self._raw_header.items():
            if re.search(">", key):  # initial dicts for modules
                if key.startswith("Ext. VI 1>"):
                    # deal with external VI attributes
                    # Use proper string slicing instead of strip
                    module_name = key[len("Ext. VI 1>") :]
                    if ">" in module_name:
                        modules.append(module_name.split(">")[0])
                    else:
                        modules.append(module_name)
                else:  # deal with the internal modules
                    modules.append(key.split(">")[0])
            else:
                header.update({key: value.strip("\r\n").strip('"')})

        modules = Counter(modules)

        for module, count in modules.items():
            if count == 1:
                for key, value in self._raw_header.items():
                    # Handle the strip issue properly
                    if key.startswith("Ext. VI 1>"):
                        check_key = key[len("Ext. VI 1>") :]
                    else:
                        check_key = key

                    if module == check_key.split(">")[0]:
                        header.update({module: value.strip("\r\n")})
            else:
                header[module] = {}
                for key, value in self._raw_header.items():
                    # Handle the strip issue properly
                    if key.startswith("Ext. VI 1>"):
                        check_key = key[len("Ext. VI 1>") :]
                    else:
                        check_key = key

                    if module == check_key.split(">")[0]:
                        header[module].update(
                            {key.split(">")[-1]: value.strip("\r\n").strip('"')},
                        )

        if "Bias Spectroscopy" in self._raw_header:
            for key, value in header["Bias Spectroscopy"].items():
                if "MultiLine Settings" in key:
                    header["Bias Spectroscopy"].update(
                        {
                            "MultiLine Settings": pd.DataFrame(
                                np.array(
                                    [
                                        list(map(float, row.split(",")))
                                        for row in value.split(";")
                                    ],
                                ),
                                columns=key.split(":")[-1].split(","),
                            ),
                        },
                    )
                    header["Bias Spectroscopy"].pop(key)
                    break
        return header

    def _reform_3ds_data(self) -> tuple[pd.DataFrame, np.ndarray]:
        """Reformats raw 3D spectroscopy (3DS) data into structured parameters and grid.

        This method processes the raw 3DS data by extracting parameters and
        spectroscopic data into separate structured formats for easier analysis.
        """
        self._ensure_header_parsed()

        pts_per_chan = int(self._header["Points"]) if "Points" in self._header else 0

        param_length = int(self._header.get("# Parameters (4 byte)", 0))
        data_length = int(int(self._header.get("Experiment size (bytes)", 0)) / 4)

        block_size = param_length + data_length

        if "Grid dim" in self._header:
            grid_dim = self._parse_grid_dim(self._header["Grid dim"])
        else:
            grid_dim = (0, 0)

        if "Channels" in self._header:
            channels = self._header["Channels"].split(";")
        else:
            channels = []

        total_pixels = grid_dim[0] * grid_dim[1]
        total_pts = block_size * total_pixels

        grid = np.empty((total_pixels, len(channels), pts_per_chan))

        fixed_params = self._header.get("Fixed parameters", "").split(";")
        exp_params = self._header.get("Experiment parameters", "").split(";")
        param_columns = fixed_params + exp_params

        params = pd.DataFrame(
            index=range(total_pixels),
            columns=param_columns,
        )

        if total_pts > self._raw_data.shape[0]:
            data = np.concatenate(
                [
                    self._raw_data,
                    np.full(max(0, total_pts - len(self._raw_data)), np.nan),
                ],
            )
        else:
            data = self._raw_data

        for i in range(0, len(data), block_size):
            n = i // block_size
            block = data[i : i + block_size]
            params.loc[n] = block[:param_length]
            grid[n] = block[param_length:block_size].reshape(
                (len(channels), pts_per_chan),
            )

        return params, grid

    def _ensure_header_parsed(self) -> None:
        """Ensure the header is parsed and cached.

        This method triggers parsing of the raw header into a structured
        dictionary if it hasn't been done yet. It's called automatically
        by the `header` property and other methods that depend on parsed header data.
        """
        if self._header is None:
            if self.file_type == "sxm":
                self._header = self._reform_sxm_header()
            elif self.file_type == "dat":
                self._header = self._reform_dat_header()
            elif self.file_type == "3ds":
                self._header = self._reform_3ds_header()

    def _ensure_data_parsed(self) -> None:
        """Ensure the data is parsed and cached.

        This method triggers parsing of the raw data into its final structured
        format (NumPy array or DataFrame) if it hasn't been done yet. It's called
        automatically by the `data` and `parameters` properties.
        """
        if self._data is None:
            if self.file_type == "sxm":
                self._ensure_header_parsed()
                self._data = self._reform_sxm_data()
            elif self.file_type == "dat":
                self._data = self._raw_data
            elif self.file_type == "3ds":
                self._ensure_header_parsed()
                self._parameters, self._data = self._reform_3ds_data()

    def _extract_ext_vi(self, key: str) -> str:
        """Extract module name from a key that may contain 'Ext. VI 1>' prefix."""
        if key.startswith("Ext. VI 1>"):
            module_part = key[len("Ext. VI 1>") :]
            return module_part.split(">")[0] if ">" in module_part else module_part
        else:
            return key.split(">")[0]

    def _process_modules(self, header: dict, modules: Counter) -> None:
        """Process modules based on their count (single vs multiple attributes)."""
        for module, count in modules.items():
            if count == 1:
                self._process_single_module(header, module)
            else:
                self._process_multi_module(header, module)

    def _process_single_module(self, header: dict, module: str) -> None:
        """Process a module with only one attribute."""
        for key, value in self._raw_header.items():
            check_key = (
                key[len("Ext. VI 1>") :] if key.startswith("Ext. VI 1>") else key
            )
            if check_key.startswith(module):
                header.update({module: value.strip("\n")})
                break

    def _process_multi_module(self, header: dict, module: str) -> None:
        """Process a module with multiple attributes."""
        header[module] = {}
        for key, value in self._raw_header.items():
            check_key = (
                key[len("Ext. VI 1>") :] if key.startswith("Ext. VI 1>") else key
            )
            if check_key.startswith(module):
                header[module].update({key.split(">")[-1]: value.strip("\n")})

    @property
    def header(self) -> dict:
        """Get the processed header dictionary from the loaded file.

        Returns:
            dict: The reformatted header data containing metadata and parameters
            extracted from the original file header.

        """
        self._ensure_header_parsed()
        return self._header

    @property
    def data(self) -> np.ndarray | pd.DataFrame | None:
        """Get the processed data from the loaded file.

        Returns:
            np.ndarray | pd.DataFrame | None: The processed data depending on file type:
                - For .sxm files: 2D numpy array of scan data
                - For .dat files: pandas DataFrame of tabular data
                - For .3ds files: 3D numpy array of spectroscopy data
                - None if not yet parsed or unavailable

        """
        self._ensure_data_parsed()
        return self._data

    @property
    def parameters(self) -> pd.DataFrame | None:
        """Get the processed parameters DataFrame from the loaded file.

        Returns:
            pd.DataFrame | None: The parameters DataFrame for 3DS files,
            or None if not applicable or not yet parsed.

        """
        self._ensure_data_parsed()
        return self._parameters

    @property
    def pts_per_chan(self) -> int:
        """Number of data points per channel in .3ds files."""
        if self.file_type != "3ds":
            error_msg = "pts_per_chan only available for .3ds files"
            raise AttributeError(error_msg)

        if self._header is not None and "Points" in self._header:
            return int(self._header["Points"])
        if self._raw_header and "Points" in self._raw_header:
            return int(self._raw_header["Points"])
        return 0

    @staticmethod
    def _parse_grid_dim(grid_str: str) -> tuple[int, int]:
        """Parse grid dimension string into (width, height) tuple.

        Handles formats like '10x20', '10 x 20', or malformed strings.
        Returns (0, 0) for invalid inputs.

        Args:
            grid_str: Grid dimension string from .3ds header (e.g., "10x20")

        Returns:
            Tuple of two integers (width, height). Returns (0, 0) on failure.

        """
        expected_parts = 2
        try:
            parts = grid_str.replace(" ", "").split("x")
            return (
                tuple(int(x) for x in parts) if len(parts) == expected_parts else (0, 0)
            )
        except (ValueError, AttributeError):
            return (0, 0)

    @property
    def pixels(self) -> tuple:
        """Get the pixel dimensions of the scan data."""
        if self._pixels is None:
            if self.file_type == "sxm":
                if self._header is not None:
                    self._pixels = tuple(
                        map(int, self._header.get("SCAN_PIXELS", "0 0").split()),
                    )
                elif self._raw_header and "SCAN_PIXELS" in self._raw_header:
                    self._pixels = tuple(
                        map(int, self._raw_header["SCAN_PIXELS"].split()),
                    )
                else:
                    self._pixels = (0, 0)
            elif self.file_type == "dat":
                self._pixels = (1, 0)
            elif self.file_type == "3ds":
                if self._header is not None:
                    grid_str = self._header.get("Grid dim", "0x0")
                elif self._raw_header and "Grid dim" in self._raw_header:
                    grid_str = self._raw_header["Grid dim"]
                else:
                    grid_str = "0x0"
                self._pixels = self._parse_grid_dim(grid_str)
            else:
                self._pixels = (0, 0)
        return self._pixels

    @property
    def range(self) -> tuple[float, float]:
        """Get the scan range (width, height) from .sxm files.

        Returns:
            tuple[float, float]: The scan range in nanometers as (width, height).

        Raises:
            AttributeError: If called on non-.sxm files.

        """
        if self.file_type != "sxm":
            error_msg = "range only available for .sxm files"
            raise AttributeError(error_msg)
        self._ensure_header_parsed()
        return tuple(map(float, self._header.get("SCAN_RANGE", "0 0").split()))

    @property
    def center(self) -> tuple[float, float]:
        """Get the scan center offset from .sxm files.

        Returns:
            tuple[float, float]: The scan center offset in nanometers as (x, y).

        Raises:
            AttributeError: If called on non-.sxm files.

        """
        if self.file_type != "sxm":
            error_msg = "center only available for .sxm files"
            raise AttributeError(error_msg)
        self._ensure_header_parsed()
        return tuple(map(float, self._header.get("SCAN_OFFSET", "0 0").split()))

    @property
    def frame_angle(self) -> float:
        """Get the scan frame angle from .sxm files.

        Returns:
            float: The scan angle in degrees from the header.

        Raises:
            AttributeError: If called on non-.sxm files.

        """
        if self.file_type != "sxm":
            error_msg = "frame_angle only available for .sxm files"
            raise AttributeError(error_msg)
        self._ensure_header_parsed()
        return float(self._header.get("SCAN_ANGLE", 0))

    @property
    def dir(self) -> bool:
        """Downward --> False, upward --> True."""
        if self.file_type != "sxm":
            error_msg = "dir only available for .sxm files"
            raise AttributeError(error_msg)
        self._ensure_header_parsed()
        return self._header.get("SCAN_DIR") == "up"

    @property
    def bias(self) -> float:
        """Get the bias value from .sxm files.

        Returns:
            float: The bias value from the header.

        Raises:
            AttributeError: If called on non-.sxm files.

        """
        if self.file_type != "sxm":
            error_msg = "bias only available for .sxm files"
            raise AttributeError(error_msg)
        self._ensure_header_parsed()
        return float(self._header.get("BIAS", 0))

    @property
    def setpoint(self) -> float:
        """Get the setpoint value from Z-CONTROLLER section in .sxm files.

        Returns:
            float: The setpoint value from the Z-CONTROLLER section.

        Raises:
            AttributeError: If called on non-.sxm files.

        """
        if self.file_type != "sxm":
            error_msg = "setpoint only available for .sxm files"
            raise AttributeError(error_msg)
        self._ensure_header_parsed()
        z_controller = self._header.get("Z-CONTROLLER")
        if z_controller is not None and "Setpoint" in z_controller.columns:
            return float(z_controller["Setpoint"].iloc[0].split()[0])
        return 0.0

    @property
    def channels(self) -> list[str]:
        """Get the list of channel names from the loaded file.

        Returns:
            list[str]: List of channel names based on file type:
                - For .sxm files: Extracts from DATA_INFO section
                - For .3ds files: Splits Channels header by semicolon
                - For .dat files: Returns DataFrame column names
                - For unsupported types: Returns empty list

        """
        channel_getters = {
            "sxm": self._get_sxm_channels,
            "3ds": self._get_3ds_channels,
            "dat": self._get_dat_channels,
        }

        getter = channel_getters.get(self.file_type)
        if getter:
            return getter()
        return []

    def _get_sxm_channels(self) -> list[str]:
        """Extract channel names from SXM file format."""
        if self._header is not None and "DATA_INFO" in self._header:
            return self._header["DATA_INFO"]["Name"].tolist()
        if self._raw_header and "DATA_INFO" in self._raw_header:
            try:
                df = pd.DataFrame(
                    [
                        row.split("\t")
                        for row in self._raw_header["DATA_INFO"].split("\n")
                    ],
                )
                df.columns = df.iloc[0]
                return df[1:]["Name"].tolist()
            except (KeyError, IndexError, ValueError):
                return []
        return []

    def _get_3ds_channels(self) -> list[str]:
        """Extract channel names from 3DS file format."""
        if self._header is not None and "Channels" in self._header:
            return self._header["Channels"].split(";")
        if self._raw_header and "Channels" in self._raw_header:
            return self._raw_header["Channels"].split(";")
        return []

    def _get_dat_channels(self) -> list[str]:
        """Extract channel names from DAT file format."""
        if self._data is not None:
            return self._data.columns.tolist()
        if self._raw_data is not None:
            return self._raw_data.columns.tolist()
        return []
