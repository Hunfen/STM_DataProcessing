# -*- coding: utf-8 -*-
"""
Python module that helps read the Nanonis files.
"""
__all__ = []

import os
import numpy as np
import re


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
            self.data = self.__read_sxm_data__(f_path, self.header)
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
                decoded_line = line.decode(encoding="utf-8", errors="replace").strip(' ')
                if re.match(":SCANIT_END:", decoded_line):
                    break
                if re.match(":.+:", decoded_line):
                    entry = decoded_line.strip('\n').strip(':')
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
        return header

    def __read_sxm_data__(self, f_path: str, header):
        data = np.empty((1, 1))
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
        return raw_header

    def __reform_3ds_header__(self, raw_header):
        header = {}
        return header

    def __read_3ds_data__(self, f_path: str, header):
        data = np.empty((1, 1))
        return data
