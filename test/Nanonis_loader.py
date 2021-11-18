# -*- coding: utf-8 -*-
import os
import re

import numpy as np


def loader(f_path: str):
    """To load the data files created by Nanonis SPM controller. 
    
    Args:
        f_path : Path to the data file.
    
    Returns:
        {
            __Nanonis_sxm__ : if file type is .sxm.
            __Nanonis_dat__ : if file type is .dat.
            __Nanonis_3ds__ : if file type is .3ds.
        }
    
    Raises:
        KeyError : If file type is still not supported.

    """
    switch = {
        '.sxm': lambda x: __Nanonis_sxm__(x),
        # '.dat': lambda x: __Nanonis_dat__(x),
        # '.3ds': lambda x: __Nanonis_3ds__(x)
    }
    try:
        return switch[os.path.splitext(f_path)[1]](f_path)
    except KeyError:
        print('File type not supported.')


class __Nanonis_sxm__:
    """[summary]

    Returns:
        [type]: [description]
    """
    file_path: str
    fname: str
    raw_header: dict
    header: dict

    def __init__(self, f_path: str) -> None:
        self.file_path = os.path.split(f_path)[0]
        self.fname = os.path.split(f_path)[1]
        self.__sxm_header_reform__(self.raw_header)

    def __sxm_header_reader__(self, f_path: str) -> None:
        """[summary]

        Args:
            f_path (str): [description]

        Returns:
            dict: [description]
        """
        entry: str = ''
        contents: str = ''
        raw_header: dict = {}
        with open(f_path, 'rb') as f:  # read the .sxm file
            header_end = False
            while not header_end:
                line = f.readline().decode(encoding='utf-8', errors='replace')
                # if reach the end of header
                if re.match(':SCANIT_END:\n', line):
                    header_end = True
                # ':.+:' is the regex of the Nanonis .sxm file header entry.
                elif re.match(':.+:', line):
                    entry = line[1:-2]  # Read header_entry
                    contents = ''  # Clear contents
                else:  # Load entries & corresponding parameters into pre-defined dict
                    contents += line
                    raw_header[entry] = contents.strip('\n')  # remove EOL
        self.raw_header = raw_header

    def __sxm_header_reform__(self, raw_header: dict) -> None:
        """[summary]

        Args:
            raw_header (dict): [description]
        """

        scan_info_float: list[str] = [
            'ACQ_TIME', 'BIAS', 'Scan>lines', 'Scan>pixels/line',
            'Scan>speed backw. (m/s)', 'Scan>speed forw. (m/s)'
        ]
        scan_info_str: list[str] = [
            'COMMENT', 'REC_DATE', 'REC_TIME', 'SCAN_DIR', 'SCAN_FILE'
        ]
        # table: list[str] = ['DATA_INFO', 'Scan>Scanfield', 'Z-CONTROLLER']
        scan_field_key: list[str] = [
            'X_OFFSET', 'Y_OFFSET', 'X_RANGE', 'Y_RANGE', 'ANGLE'
        ] # The order of scan_field_key should not be changed
        trash_bin: list[str] = [
            'NANONIS_VERSION', 'REC_TEMP', 'SCANIT_TYPE', 'SCAN_ANGLE',
            'SCAN_OFFSET', 'SCAN_PIXELS', 'SCAN_RANGE', 'SCAN_TIME',
            'Scan>channels'
        ]

        header: dict = {}
        entries: list = list(raw_header.keys())
        
        for i in enumerate(entries):
            if i[1] in scan_info_float:
                header[i[1]] = float(raw_header[i[1]])
            elif i[1] in scan_info_str:
                header[i[1]] = raw_header[i[1]].strip(' ')
            elif i[1] == 'Z-CONTROLLER':
                z_controller: dict = {}
                z_controller_config = raw_header['Z-CONTROLLER'].split('\n')[0].strip('\t').split('\t')
                z_controller_configVar = raw_header['Z-CONTROLLER'].split('\n')[1].strip('\t').split('\t')
                for j, s in enumerate(z_controller_config):
                    z_controller[s] = z_controller_configVar[j]
                header[i[1]] = z_controller
            elif i[1] == 'DATA_INFO':
                data_info: dict = {}
                config = raw_header['DATA_INFO'].split('\n')[0].strip('\t').split('\t')
                
            


# class __Nanonis_dat__:
#     """

#     """

#     file_path: str
#     fname: str

#     def __init__(self, f_path: str) -> None:
#         self.file_path = os.path.split(f_path)[0]
#         self.fname = os.path.split(f_path)[1]

# class __Nanonis_3ds__:
#     """[summary]
#     """
#     file_path: str
#     fname: str

#     def __init__(self, f_path: str) -> None:
#         self.file_path = os.path.split(f_path)[0]
#         self.fname = os.path.split(f_path)[1]
