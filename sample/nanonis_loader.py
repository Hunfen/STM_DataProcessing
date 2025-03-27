# -*- coding: utf-8 -*-
"""
Python module that helps read the Nanonis files.
"""

__all__ = []


import os
import re
from collections import Counter
from . import np, pd


class NanonisFileLoader:
    """_summary_"""
# TODO：
# 1. NanonisFileLoader类初始化后不直接对header和data进行处理，
# 而是在需要的时候对header和data进行reformat。
# 2. I(V), dI/dV(V) 插值平均

    def __init__(self, f_path: str) -> None:
        self.file_path, self.fname = os.path.split(f_path)
        self._header = {}
        self._data = None
        self._parameters = None
        self._pixels = None

        if f_path.endswith(".sxm"):
            self.file_type = "sxm"
            self.raw_header, self.__data_1d = self.__sxm_loader__(f_path)
            self._header = self.__reform_sxm_header__()
            # ---------------------------------------------------------------------------
            # General scanning information quick access
            self.range = tuple(map(float, self.header["SCAN_RANGE"].split()))
            self.center = tuple(map(float, self.header["SCAN_OFFSET"].split()))
            self.frame_angle = float(self.header["SCAN_ANGLE"])
            # downward --> false, upward --> true
            self.dir = self.header["SCAN_DIR"] == "up"
            self.bias = float(self.header["BIAS"])
            self.setpoint = float(
                self.header["Z-CONTROLLER"]["Setpoint"][0].split()[0])
            self.channels: list = self.header["DATA_INFO"]["Name"].tolist()
            # --------------------------------------------------------------------------
            self._data = self.__reform_sxm_data__()

        elif f_path.endswith(".dat"):
            self.file_type = "dat"
            self.raw_header, self._data = self.__dat_loader__(f_path)
            self._header = self.__reform_dat_header__()

        elif f_path.endswith(".3ds"):
            self.file_type = "3ds"
            self.raw_header, self.__data_1d = self.__3ds_loader__(f_path)
            self._header = self.__reform_3ds_header__()
            # --------------------------------------------------------------------------
            # General scanning information quick access

            self.param_length = int(
                self.header.get("# Parameters (4 byte)", 0))
            self.data_length = int(
                int(self.header.get("Experiment size (bytes)", 0)) / 4
            )
            self.channels = self.header["Channels"].split(";")
            self.pts_per_chan = int(self.header["Points"])
            # --------------------------------------------------------------------------
            self._parameters, self._data = self.__reform_3ds_data__()
        else:
            raise ValueError(f"Unsupported file type: {f_path}")

    def __sxm_loader__(self, f_path: str):
        """
        Load and parse .sxm file, extracting header and binary data.

        Reads header until ":SCANIT_END:" marker, then extracts binary
        data as big-endian floats. Header entries start with ":[key]:"
        followed by content lines.

        Args:
            f_path: Path to .sxm file as string.

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
        entry, contents, contents_updated = "", "", False
        with open(f_path, "rb") as f:
            # Read and parse the header
            for line in f:
                decoded_line = line.decode(encoding="utf-8", errors="replace")
                if ":SCANIT_END:" in decoded_line:
                    raw_header.update({entry: contents})
                    break
                if re.fullmatch(r":.*:\n", decoded_line):
                    # in the key line
                    if contents_updated:
                        # encounter the next key,
                        # update last {key: contents}
                        # & initialize key and contents
                        raw_header.update({entry: contents.rstrip("\n")})
                    entry = decoded_line.rstrip("\n").strip(":")
                    contents, contents_updated = "", False
                else:  # in the contents line, update contents
                    contents += decoded_line
                    contents_updated = True

            # Skip two lines and move the file pointer
            for _ in range(2):
                f.readline()
            f.seek(2, 1)

            # Read the binary data
            data = np.fromfile(f, dtype=">f")
        return raw_header, data

    def __reform_sxm_header__(self):
        """
        Reformats the raw header data from a .sxm file
        into a structured dictionary.

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

        for key, value in self.raw_header.items():
            if re.search(">", key):  # initial dicts for modules
                if key.startswith("Ext. VI 1>"):
                    # deal with external VI attributes
                    modules.append(key.strip("Ext. VI 1>").split(">")[0])
                else:  # deal with the internal modules
                    modules.append(key.split(">")[0])
            else:
                header.update({key: value})

        # count the number of attributes in every module
        modules = Counter(modules)

        for key in ["Z-CONTROLLER", "DATA_INFO"]:
            df = pd.DataFrame(
                [row.split("\t") for row in self.raw_header[key].split("\n")]
            )
            df.columns = df.iloc[0]
            header[key] = df[1:].reset_index(drop=True).dropna(how="any")

        for module, count in modules.items():
            if count == 1:
                for key, value in self.raw_header.items():
                    if key.strip("Ext. VI 1>").startswith(module):
                        header.update({module: value.strip("\n")})
            else:
                header[module] = {}
                for key, value in self.raw_header.items():
                    if key.strip("Ext. VI 1>").startswith(module):
                        header[module].update(
                            {key.split(">")[-1]: value.strip("\n")})
                        # temp_dict[key.split(">")[-1]] = [value.strip("\n")]
                    else:
                        continue

        return header

    def __reform_sxm_data__(self):
        """
        Reformats the SXM data by reshaping and handling backward and upward
        scanning directions. The data is reshaped to match the number of
        channels and pixels. Backward scans are flipped horizontally, and
        upward scans are flipped vertically if the direction flag is set.

        Returns:
            numpy.ndarray: Reformatted SXM data with corrected scanning
                          directions.
        """
        total_pts = len(self.channels) * 2 * self.pixels[0] * self.pixels[1]
        if total_pts > len(self.__data_1d):
            data = np.concatenate(
                [
                    self.__data_1d,
                    np.full(max(0, total_pts - len(self.__data_1d)), np.nan),
                ]
            )
        data = self.__data_1d.reshape((len(self.channels) * 2, *self.pixels))
        for i in range(data.shape[0]):
            if i % 2 != 0:
                # deal with backward scanning
                data[i] = np.fliplr(data[i])
            if self.dir:
                # deal with upward scanning
                data[i] = np.flipud(data[i])

        return data

    def __dat_loader__(self, f_path: str) -> tuple[dict, pd.DataFrame]:
        """
        Loads Nanonis .dat file, parsing header and data sections.

        Reads binary file to handle line endings precisely. Header entries
        use format: key<tab>value<tab>\\r\\n. Data section starts after
        [DATA] marker with tab-delimited columns.

        Args:
            f_path (str): Path to .dat file

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

        with open(f_path, "rb") as f:
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
                        key, value_buffer = decoded_line.rstrip(
                            "\t\r\n").split("\t", 1)
                    else:
                        value_buffer += decoded_line.rstrip("\t\r\n")
                    raw_header.update({key: value_buffer})
                    value_buffer = ""
                else:
                    if "\t" in decoded_line:
                        key, value_buffer = decoded_line.split("\t", 1)
                    else:
                        value_buffer += decoded_line
            for line in f:
                decoded_line = line.decode("utf-8", errors="replace")
                if decoded_line.rstrip("\r\n"):
                    data_lines.append(decoded_line.rstrip("\r\n").split("\t"))
        data = pd.DataFrame(data_lines, columns=columns).apply(
            pd.to_numeric, errors="coerce"
        )
        return raw_header, data

    def __reform_dat_header__(self):
        header = {}
        modules = []

        for key, value in self.raw_header.items():
            if re.search(">", key):  # initial dicts for modules
                if key.startswith("Ext. VI 1>"):
                    # deal with external VI attributes
                    modules.append(key.strip("Ext. VI 1>").split(">")[0])
                else:  # deal with the internal modules
                    modules.append(key.split(">")[0])
            else:
                header.update({key: value.strip("\n")})

        modules = Counter(modules)

        for module, count in modules.items():
            if count == 1:
                for key, value in self.raw_header.items():
                    if module == key.strip("Ext. VI 1>").split(">")[0]:
                        header.update({module: value.strip("\r\n")})
            else:
                # continue
                header[module] = {}
                for key, value in self.raw_header.items():
                    if module == key.strip("Ext. VI 1>").split(">")[0]:
                        header[module].update(
                            {key.split(">")[-1]: value.strip("\r\n").strip('"')
                             }
                        )
                    else:
                        continue

        for key, value in header["Bias Spectroscopy"].items():
            if "MultiLine Settings" in key:
                header["Bias Spectroscopy"].update(
                    {
                        "MultiLine Settings": pd.DataFrame(
                            np.array(
                                [
                                    list(map(float, row.split(",")))
                                    for row in value.split(";")
                                ]
                            ),
                            columns=key.split(":")[-1].split(","),
                        )
                    }
                )
                header["Bias Spectroscopy"].pop(key)
                break
        return header

    def __3ds_loader__(self, f_path: str):
        """
        Loads Nanonis .3ds file, parsing header and binary data.

        Reads file in binary mode to handle header and data precisely.
        Header ends at ':HEADER_END:' marker. Data is read as big-endian
        floats after header.

        Args:
            f_path (str): Path to .3ds file

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
        with open(f_path, "rb") as f:
            for line in f:
                decoded_line = line.decode(encoding="utf-8", errors="replace")
                if ":HEADER_END:" in decoded_line:
                    # b'\x0d\x0a' is following the ":HEADER_END:"
                    # f.readline() will automaticly skip them
                    break
                if decoded_line.endswith("\r\n"):
                    # Update key and value when encounter '\r\n'
                    if "=" in decoded_line.strip("\r\n"):
                        key, value_buffer = decoded_line.strip(
                            "\r\n").split("=", 1)
                    else:
                        value_buffer += decoded_line.rstrip("\t\r\n")
                    raw_header.update({key: value_buffer})
                    value_buffer = ""
                else:
                    if "=" in decoded_line:
                        key, value_buffer = decoded_line.split("=", 1)
                    else:
                        value_buffer += decoded_line
            data_1d = np.fromfile(f, dtype=">f")
        return raw_header, data_1d

    def __reform_3ds_header__(self):
        """
        Reformats the raw header data into a structured dictionary.

        Parses the raw header, organizes module-specific data, and converts
        specific fields (e.g., 'MultiLine Settings') into appropriate formats.

        Returns:
            dict: Structured header with module-specific data.
        """
        header = {}
        modules = []
        for key, value in self.raw_header.items():
            if re.search(">", key):  # initial dicts for modules
                if key.startswith("Ext. VI 1>"):
                    # deal with external VI attributes
                    modules.append(key.strip("Ext. VI 1>").split(">")[0])
                else:  # deal with the internal modules
                    modules.append(key.split(">")[0])
            else:
                header.update({key: value.strip("\r\n").strip('"')})

        modules = Counter(modules)

        for module, count in modules.items():
            if count == 1:
                for key, value in self.raw_header.items():
                    if module == key.strip("Ext. VI 1>").split(">")[0]:
                        header.update({module: value.strip("\r\n")})
            else:
                # continue
                header[module] = {}
                for key, value in self.raw_header.items():
                    if module == key.strip("Ext. VI 1>").split(">")[0]:
                        header[module].update(
                            {key.split(">")[-1]: value.strip("\r\n").strip('"')
                             }
                        )
                    else:
                        continue

        if 'Bias Spectroscopy' in self.raw_header.keys():
            for key, value in header["Bias Spectroscopy"].items():
                if "MultiLine Settings" in key:
                    header["Bias Spectroscopy"].update(
                        {
                            "MultiLine Settings": pd.DataFrame(
                                np.array(
                                    [
                                        list(map(float, row.split(",")))
                                        for row in value.split(";")
                                    ]
                                ),
                                columns=key.split(":")[-1].split(","),
                            )
                        }
                    )
                    header["Bias Spectroscopy"].pop(key)
                    break
        return header

    def __reform_3ds_data__(self):
        """
        Reformats raw 3D spectroscopy (3DS) data into structured parameters
        and grid.

        Processes raw 1D data into structured format by:
        1. Calculating total data size from header metadata.
        2. Reshaping into grid (pixels × channels × points per channel).
        3. Extracting experiment parameters into DataFrame.

        Returns:
            tuple: Contains:
                - params (pd.DataFrame): DataFrame with pixel parameters.
                    Columns from 'Fixed parameters' and 'Experiment parameters'
                - grid (np.ndarray): 3D array of shape
                    (pixels[0]*pixels[1], len(channels), pts_per_chan).

        Note:
            - Pads raw data with NaN if shorter than expected.
            - Assumes raw data in self.__data_1d and metadata in self.header.
        """
        param_length = int(self.header["# Parameters (4 byte)"])
        data_length = int(int(self.header["Experiment size (bytes)"]) / 4)

        block_size = param_length + data_length
        total_pts = block_size * self.pixels[0] * self.pixels[1]
        grid = np.empty(
            (self.pixels[0] * self.pixels[1],
             len(self.channels), self.pts_per_chan)
        )
        params = pd.DataFrame(
            index=range(self.pixels[0] * self.pixels[1]),
            columns=(
                self.header["Fixed parameters"].split(";")
                + self.header["Experiment parameters"].split(";")
            ),
        )

        if total_pts > self.__data_1d.shape[0]:
            data = np.concatenate(
                [
                    self.__data_1d,
                    np.full(max(0, total_pts - len(self.__data_1d)), np.nan),
                ]
            )
        else:
            data = self.__data_1d

        for i in range(0, len(data), block_size):
            n = i // block_size
            block = data[i: i + block_size]
            params.loc[n] = block[:param_length]
            grid[n] = block[param_length:block_size].reshape(
                (len(self.channels), self.pts_per_chan)
            )

        return params, grid

    @property
    def pixels(self) -> tuple:
        """Get the pixel dimensions of the scan data.

        Returns:
            tuple: A tuple of two integers representing the pixel dimensions
            (width, height). For 'sxm' files, reads from 'SCAN_PIXELS' in
            the header. For 'dat' files, returns (1, 0). For '3ds' files,
            parses 'Grid dim' from the header. If parsing fails or the file
            type is unknown, returns (0, 0).
        """
        if self._pixels is None:
            if self.file_type == "sxm":
                self._pixels = tuple(
                    map(int, self.header.get("SCAN_PIXELS", "0 0").split())
                )
            elif self.file_type == "dat":
                self._pixels = (1, 0)
            elif self.file_type == "3ds":
                self._pixels = tuple(
                    int(item)
                    for item in self.header.get("Grid dim", "0x0")
                    .replace(" ", "")
                    .split("x")
                )
            else:
                self._pixels = (0, 0)
        return self._pixels

    @property
    def header(self) -> dict:
        """Return the header dictionary.

        Returns:
            dict: The stored header information.
        """
        return self._header

    @property
    def data(self) -> np.ndarray | pd.DataFrame | None:
        """Return the stored data.

        Returns:
            np.ndarray | pd.DataFrame | None:
                The stored data, which can be a NumPy array,
                a pandas DataFrame, or None if no data is loaded.
        """
        return self._data

    @property
    def parameters(self):
        """Return the stored parameters dictionary.

        Returns:
            dict: The dictionary containing all measurement parameters.
        """
        return self._parameters
