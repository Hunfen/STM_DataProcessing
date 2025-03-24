# -*- coding: utf-8 -*-
"""
Python module that helps read the Nanonis files.
"""
__all__ = []

import os
import re
from collections import Counter
import numpy as np
import pandas as pd


class NanonisFileLoader:
    """_summary_"""

    def __init__(self, f_path: str) -> None:
        self.file_path = os.path.split(f_path)[0]
        self.fname = os.path.split(f_path)[1]
        self.header = {}
        self.data = None
        self.channel_dir = []

        if f_path.endswith(".sxm"):
            self.file_type = "sxm"
            self.raw_header, self.__data_stream = self.__sxm_loader__(f_path)
            self.header = self.__reform_sxm_header__()
            # ---------------------------------------------------------------------------
            # General scanning information quick access
            self.pixels = tuple(map(int, self.header["SCAN_PIXELS"].split()))
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
            self.data = self.__reform_sxm_data__()

        elif f_path.endswith(".dat"):
            self.file_type = "dat"
            self.raw_header = self.__read_dat_header__(f_path)
            self.header = self.__reform_dat_header__(self.raw_header)
            self.data = self.__read_dat_data__(f_path, self.header)
        elif f_path.endswith(".3ds"):
            self.file_path = "3ds"
            self.raw_header, self.__data_1d = self.__3ds_loader__(f_path)
            self.header = self.__reform_3ds_header__()
            # --------------------------------------------------------------------------
            # General scanning information quick access
            self.pixels = tuple(int(item)
                                for item in self.header.get('Grid dim', '0x0').
                                replace(" ", "").split('x'))
            self.param_length = int(
                self.header.get('# Parameters (4 byte)', 0))
            self.data_length = int(
                int(self.header.get('Experiment size (bytes)', 0)) / 4)
            self.channels = self.header['Channels'].split(';')
            self.pts_per_chan = int(self.header['Points'])
            # --------------------------------------------------------------------------
            self.parameters, self.data = self.__reform_3ds_data__()
        else:
            raise ValueError(f"Unsupported file type: {f_path}")

    def __sxm_loader__(self, f_path: str):
        """
        Loads and parses a .sxm file, extracting the header and binary data.

        Args:
            f_path (str): The file path to the .sxm file.

        Returns:
            tuple: A tuple containing two elements:
                - raw_header (dict): parsed header information.
                - 1d data (numpy.ndarray):binary data from the file.

        The method reads the header line by line, stopping when it encounters
        the ":SCANIT_END:" marker.
        It then skips two lines and moves the file pointer to the start of the
        binary data, which is read
        as a NumPy array of big-endian floats.
        """
        raw_header = {}
        with open(f_path, "rb") as f:
            # Read and parse the header
            for line in f:
                decoded_line = line.decode(encoding="utf-8", errors="replace").strip(
                    " "
                )
                if ":SCANIT_END:" in decoded_line:
                    break
                if re.match(":.+:", decoded_line):
                    entry = decoded_line.strip("\n").strip(":")
                    contents = ""
                else:
                    if not contents:
                        contents = ""
                    contents += decoded_line
                    raw_header[entry] = contents

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
                header[key] = value.strip("\n")

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
                        header[module] = value.strip("\n")
            else:
                temp_dict = {}
                for key, value in self.raw_header.items():
                    if key.strip("Ext. VI 1>").startswith(module):
                        temp_dict[key.split(">")[-1]] = [value.strip("\n")]
                    else:
                        continue
                header[module] = pd.DataFrame(temp_dict)

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
        data = self.__data_stream.reshape(
            (len(self.channels) * 2, *self.pixels))
        for i in range(data.shape[0]):
            if i % 2 != 0:
                # deal with backward scanning
                data[i] = np.fliplr(data[i])
            if self.dir:
                # deal with upward scanning
                data[i] = np.flipud(data[i])

        return data

    def __read_dat_header__(self, f_path: str):
        raw_header = {}
        return raw_header

    def __reform_dat_header__(self, raw_header):
        header = {}
        return header

    def __read_dat_data__(self, f_path: str, header):
        data = np.empty((1, 1))
        return data

    def __3ds_loader__(self, f_path: str):
        """
        Loads a 3DS file and extracts its header and 1d data.

        Args:
            f_path (str): The file path to the 3DS file.

        Returns:
            tuple: A tuple containing two elements:
                - raw_header (dict): A dictionary containing the header
                information extracted from the file.
                - data_stream (numpy.ndarray): A NumPy array containing the
                1d data extracted from the file.

        Notes:
            - The header is read until the ":HEADER_END:" marker is
            encountered.
            - The data stream is read as a NumPy array with big-endian float
            data type.
        """
        raw_header = {}
        with open(f_path, "rb") as f:
            for line in f:
                decoded_line = line.decode(encoding="utf-8", errors="replace")
                if ":HEADER_END:" in decoded_line:
                    # b'\x0d\x0a' is following the ":HEADER_END:"
                    # f.readline() will automaticly skip them
                    break
                elif "=" in decoded_line:
                    raw_header.update(dict([decoded_line.split("=", 1)]))
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
                    if module == key.strip("Ext. VI 1>").split('>')[0]:
                        header.update({module: value.strip("\r\n")})
            else:
                # continue
                header[module] = {}
                for key, value in self.raw_header.items():
                    if module == key.strip("Ext. VI 1>").split('>')[0]:
                        header[module].update(
                            {key.split(">")[-1]                             : value.strip("\r\n").strip('"')}
                        )
                    else:
                        continue

        for key, value in header['Bias Spectroscopy'].items():
            if "MultiLine Settings" in key:

                header['Bias Spectroscopy'].update(
                    {
                        "MultiLine Settings": pd.DataFrame(np.array(
                            [list(map(float, row.split(",")))
                             for row in value.split(';')]
                        ), columns=key.split(':')[-1].split(","))
                    }
                )
                header['Bias Spectroscopy'].pop(key)
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
                    Columns from 'Fixed parameters' and 'Experiment parameters'.
                - grid (np.ndarray): 3D array of shape
                    (pixels[0]*pixels[1], len(channels), pts_per_chan).

        Note:
            - Pads raw data with NaN if shorter than expected.
            - Assumes raw data in self.__data_1d and metadata in self.header.
        """
        param_length = int(self.header['# Parameters (4 byte)'])
        data_length = int(int(self.header['Experiment size (bytes)']) / 4)

        block_size = param_length + data_length
        total_pts = block_size * self.pixels[0] * self.pixels[1]
        grid = np.empty(
            (self.pixels[0] * self.pixels[1], len(self.channels), self.pts_per_chan))
        params = pd.DataFrame(columns=(
            self.header['Fixed parameters'].split(
                ';') + self.header['Experiment parameters'].split(';')))

        if total_pts > self.__data_1d.shape[0]:
            data = np.concatenate([self.__data_1d, np.full(
                max(0, total_pts - len(self.__data_1d)), np.nan)])
        else:
            data = self.__data_1d

        for i in range(0, len(data), block_size):
            block = data[i: i+block_size]
            params.loc[len(params)] = block[:param_length]
            grid[i//block_size] = block[param_length:block_size].reshape(
                (len(self.channels), self.pts_per_chan))

        return params, grid
