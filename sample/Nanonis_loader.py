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
            self.raw_header = self.__read_sxm_header__(f_path)
            self.header = self.__reform_sxm_header__(self.raw_header)
            # ---------------------------------------------------------------------------
            # General scanning information quick access
            self.pixels = tuple(map(int, self.header["SCAN_PIXELS"].split()))
            self.range = tuple(map(float, self.header["SCAN_RANGE"].split()))
            self.center = tuple(map(float, self.header["SCAN_OFFSET"].split()))
            self.frame_angle = float(self.header["SCAN_ANGLE"])
            self.dir = self.header["SCAN_DIR"] == "up"
            self.bias = float(self.header["BIAS"])
            self.setpoint = float(self.header["Z-CONTROLLER"]["Setpoint"][0].split()[0])
            self.channels: list = self.header["Scan"]["channels"][0].split(";")
            # --------------------------------------------------------------------------
            self.data = self.__read_sxm_data__(f_path)

        elif f_path.endswith(".dat"):
            self.file_type = "dat"
            self.raw_header = self.__read_dat_header__(f_path)
            self.header = self.__reform_dat_header__(self.raw_header)
            self.data = self.__read_dat_data__(f_path, self.header)
        elif f_path.endswith(".3ds"):
            self.file_path = "3ds"
            self.raw_header = self.__read_3ds_header__(f_path)
            self.header = self.__reform_3ds_header__(self.raw_header)
            self.data = self.__read_3ds_data__(f_path, self.header)

    def __read_sxm_header__(self, f_path: str):
        """_summary_

        Args:
            f_path (str): _description_

        Returns:
            _type_: _description_
        """
        raw_header = {}
        decoded_line: str = ""
        with open(f_path, "rb") as f:
            for line in f:
                decoded_line = line.decode(encoding="utf-8", errors="replace").strip(
                    " "
                )
                if re.match(":SCANIT_END:", decoded_line):
                    break
                if re.match(":.+:", decoded_line):
                    entry = decoded_line.strip("\n").strip(":")
                    contents = ""
                else:
                    if not contents:
                        contents = ""
                    contents += decoded_line
                    raw_header[entry] = contents
        return raw_header

    def __reform_sxm_header__(self, raw_header):
        """_summary_

        Args:
            raw_header (_type_): _description_

        Returns:
            _type_: _description_
        """
        header = {}
        modules = []

        for key in raw_header.keys():
            if re.search(">", key):  # initial dicts for modules
                if key.startswith("Ext. VI 1>"):
                    # deal with external VI attributes
                    modules.append(key.strip("Ext. VI 1>").split(">")[0])
                else:  # deal with the internal modules
                    modules.append(key.split(">")[0])

            else:
                header[key] = raw_header[key].strip("\n")
        # count the number of attributes in every module
        modules = Counter(modules)

        for key in ["Z-CONTROLLER", "DATA_INFO"]:
            df = pd.DataFrame([row.split("\t") for row in raw_header[key].split("\n")])
            df.columns = df.iloc[0]
            header[key] = df[1:].reset_index(drop=True).dropna(how="any")

        for module, count in modules.items():
            if count == 1:
                for key, value in raw_header.items():
                    if key.strip("Ext. VI 1>").startswith(module):
                        header[module] = value.strip("\n")
            else:
                temp_dict = {}
                for key, value in raw_header.items():
                    if key.strip("Ext. VI 1>").startswith(module):
                        temp_dict[key.split(">")[-1]] = [value.strip("\n")]
                    else:
                        continue
                header[module] = pd.DataFrame(temp_dict)

        return header

    def __read_sxm_data__(self, f_path: str):
        with open(f_path, "rb") as f:
            read_all = f.read()
            offset = read_all.find(b"\x1a\x04")
            f.seek(offset + 2)
            data = np.fromfile(f, dtype=">f")
            data = data.reshape((len(self.channels) * 2, *self.pixels))
            for i in range(data.shape[0]):
                if i % 2 == 0:
                    data[i] = np.fliplr(data[i])

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

    def __read_3ds_header__(self, f_path: str):
        raw_header = {}
        with open(f_path, "rb") as f:
            for line in iter(
                lambda: f.readline().decode("utf-8", errors="replace").strip(),
                ":HEADER_END:",
            ):
                if "=" not in line:
                    continue
                entry, contents = line.split("=")
                raw_header[entry] = contents.rstrip("\r\n")

        return raw_header

    def __reform_3ds_header__(self, raw_header):
        header = {}
        return header

    def __read_3ds_data__(self, f_path: str, header):
        data = np.empty((1, 1))

        return data
