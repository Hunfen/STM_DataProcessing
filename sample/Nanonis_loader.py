# -*- coding: utf-8 -*-
"""
Python module that helps read the Nanonis files.
"""
__all__ = []

import os


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

    def __read_sxm_header__(self, f_path: str):
        raw_header = {}
        return raw_header

    def __reform_sxm_header__(self, raw_header):
        header = {}
        return header
    
